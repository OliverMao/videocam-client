import torch
import torch.nn.functional as F
from transformers import Qwen3VLModel, Qwen3VLForConditionalGeneration, Qwen3VLVisionModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast,BaseModelOutputWithDeepstackFeatures
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack

class Qwen3VLModelRepliceVision(Qwen3VLModel):
    def __init__(self,config, mode="client"):
        super().__init__(config)
        # 替换视觉模型
        self.visual = Qwen3VLVisionModelSplit(config.vision_config, mode=mode)

class  Qwen3VLVisionModelSplit(Qwen3VLVisionModel):
    def __init__(self, config, mode="client"):
        super().__init__(config)
        self.mode = mode
        self.client_layers = 1
    
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        split_layer = min(max(self.client_layers, 0), len(self.blocks))
        deepstack_feature_lists = []

        # client 先执行前几层 ViT（默认1层），并将中间 hidden states 传给 server 续跑。
        if self.mode == "client":
            for layer_num in range(split_layer):
                blk = self.blocks[layer_num]
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if layer_num in self.deepstack_visual_indexes:
                    ds_idx = self.deepstack_visual_indexes.index(layer_num)
                    deepstack_feature = self.deepstack_merger_list[ds_idx](hidden_states)
                    deepstack_feature_lists.append(deepstack_feature)

            return {
                "hidden_states": hidden_states,
                "start_layer": split_layer,
                "deepstack_features": deepstack_feature_lists,
            }

        # server 模式：按原路径执行完整视觉主干。
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )


        


class Qwen3VLModelSplit(Qwen3VLModelRepliceVision):
    def __init__(self, config, mode="client"):
        super().__init__(config, mode=mode)
        self.mode = mode


class Qwen3VLForSplitInference(Qwen3VLForConditionalGeneration):
    def __init__(self, config, mode="client"):
        super().__init__(config)
        # 替换核心模型：让整个推理链走 Qwen3VLModelSplit。
        self.model = Qwen3VLModelSplit(config, mode=mode)
        self.post_init()