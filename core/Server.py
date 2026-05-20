import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from Qwen3ModelSplit import Qwen3VLForSplitInference

DEFAULT_CHECKPOINT = '/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/jzy/models/Qwen3-VL-2B-Instruct'
FEATURE_FILE = "/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/jzy/VideoCam/backend/app/core/packet.pt"


def move_to_device(data, device):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    return data


def run_vision_tail_from_split(
    visual,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    start_layer: int,
    deepstack_prefix: list[torch.Tensor] | None = None,
):
    # 从 split 点继续：hidden_states 已包含 patch_embed + pos_embed + client 执行的前若干层 block。
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = list(deepstack_prefix) if deepstack_prefix is not None else []
    for layer_num in range(start_layer, len(visual.blocks)):
        blk = visual.blocks[layer_num]
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_num in visual.deepstack_visual_indexes:
            idx = visual.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = visual.deepstack_merger_list[idx](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    merged_hidden_states = visual.merger(hidden_states)
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        deepstack_features=deepstack_feature_lists,
    )


def split_image_features(vision_output, image_grid_thw: torch.Tensor, spatial_merge_size: int):
    split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
    image_embeds = torch.split(vision_output.pooler_output, split_sizes)

    split_deepstack = []
    for ds in vision_output.deepstack_features:
        split_deepstack.append(torch.split(ds, split_sizes))

    deepstack_by_layer = []
    for layer_chunks in split_deepstack:
        deepstack_by_layer.append(torch.cat(layer_chunks, dim=0))

    return image_embeds, deepstack_by_layer


def run_split_packet_inference(model, processor, packet: dict, max_new_tokens: int = 200) -> str:
    """执行分拆服务端的预填充与自回归生成。"""
    device = model.device

    input_ids = move_to_device(packet["input_ids"], device)
    attention_mask = move_to_device(packet.get("attention_mask"), device)
    mm_token_type_ids = move_to_device(packet.get("mm_token_type_ids"), device)
    image_grid_thw = move_to_device(packet["image_grid_thw"], device)
    vision_hidden_states = move_to_device(packet["vision_hidden_states"], device)
    split_layer = int(packet.get("split_layer", 0))
    deepstack_prefix = move_to_device(packet.get("deepstack_prefix", []), device)

    with torch.no_grad():
        vision_output = run_vision_tail_from_split(
            model.model.visual,
            vision_hidden_states,
            image_grid_thw,
            start_layer=split_layer,
            deepstack_prefix=deepstack_prefix,
        )
        image_embeds_chunks, deepstack_image_embeds = split_image_features(
            vision_output,
            image_grid_thw,
            model.model.visual.spatial_merge_size,
        )

        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        image_embeds = torch.cat(image_embeds_chunks, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]
        position_ids = model.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

        lm_outputs = model.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
        )
        hidden_states = lm_outputs.last_hidden_state
        logits = model.lm_head(hidden_states[:, -1:, :])

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        past_key_values = lm_outputs.past_key_values

        stop_tokens = [processor.tokenizer.eos_token_id]
        if hasattr(processor.tokenizer, "additional_special_tokens_ids"):
            stop_tokens.extend(processor.tokenizer.additional_special_tokens_ids)

        generated_token_ids = []
            
        token_val = next_token_id.item()
        if token_val not in stop_tokens:
            generated_token_ids.append(token_val)

        for _ in range(max_new_tokens - 1):
            if token_val in stop_tokens:
                break
                
            step_embeds = model.model.get_input_embeddings()(next_token_id)
            step_pos_ids = model.model.compute_3d_position_ids(
                input_ids=next_token_id,
                inputs_embeds=step_embeds,
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=None,
                past_key_values=past_key_values,
                mm_token_type_ids=None,
            )

            lm_outputs = model.model.language_model(
                input_ids=None,
                position_ids=step_pos_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                inputs_embeds=step_embeds,
                use_cache=True,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
            )
            logits = model.lm_head(lm_outputs.last_hidden_state[:, -1:, :])
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            past_key_values = lm_outputs.past_key_values

            token_val = next_token_id.item()
            if token_val in stop_tokens:
                break
            generated_token_ids.append(token_val)

    return processor.decode(generated_token_ids, skip_special_tokens=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--packet", default=FEATURE_FILE)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading server model from {DEFAULT_CHECKPOINT}...")
    model = Qwen3VLForSplitInference.from_pretrained(
        DEFAULT_CHECKPOINT,
        device_map="cpu" if args.cpu_only else "auto",
        trust_remote_code=True,
        mode="server",
    )
    processor = AutoProcessor.from_pretrained(DEFAULT_CHECKPOINT, trust_remote_code=True)
    device = model.device

    print("[Server] Loading packet...")
    data = torch.load(args.packet, map_location="cpu")
    
    print("[Server] Running inference...")
    result = run_split_packet_inference(model, processor, data, max_new_tokens=args.max_new_tokens)
    
    print("\n[Server] Result:")
    print(result)

    print("\n[Server] Done.")


if __name__ == "__main__":
    main()