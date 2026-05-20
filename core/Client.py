import torch
import time
from argparse import ArgumentParser
from transformers import AutoProcessor

from Qwen3ModelSplit import Qwen3VLForSplitInference

DEFAULT_IMAGE_PATH = "/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/jzy/VideoCam/backend/app/core/smoke1.png"
DEFAULT_TEXT = "图片中是否有违规行为，包括但不限于：1.吸烟 2.打架 3.着火 4.摔倒 5.赌博 6.其他违规行为。明确指出违规行为。输出如下：\n吸烟。\n如果图片中没有违规行为，输出如下：\n无。"
DEFAULT_CHECKPOINT = '/workspace/s/ddn/gemini/gemini-sharedata/space/wqmu4k88unnm/guarded_files/jzy/models/Qwen3-VL-2B-Instruct'
DEFAULT_PACKET_PATH = "packet.pt"


def _pick_tensor(inputs, key):
    value = inputs.get(key)
    if value is None:
        return None
    return value.detach().cpu()


def _profile_client_compute(visual_model, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
    patch_embed = visual_model.patch_embed

    # 1) 优先 calflops：统计完整 client 阶段（当前为 ViT 前1层）
    try:
        from calflops import calculate_flops

        # 对 split 后的视觉前向显式传入真实输入，避免 input_shape 模式误判。
        try:
            flops, macs, params = calculate_flops(
                model=visual_model,
                kwargs={"hidden_states": pixel_values, "grid_thw": image_grid_thw},
                output_as_string=False,
                print_results=False,
                print_detailed=False,
            )
        except TypeError:
            # 兼容部分版本参数名不同的情况
            flops, macs, params = calculate_flops(
                model=visual_model,
                args=[pixel_values, image_grid_thw],
                output_as_string=False,
                print_results=False,
                print_detailed=False,
            )
        return {
            "tool": "calflops",
            "scope": "client-stage",
            "flops": flops,
            "macs": float(macs),
            "params": float(params),
        }
    except Exception as e:
        print('error 无法使用calflops',e)
        pass

    # 2) 其次 thop：同样尝试统计完整 client 阶段。
    try:
        from thop import profile

        macs, params = profile(visual_model, inputs=(pixel_values, image_grid_thw), verbose=False)
        return {
            "tool": "thop",
            "scope": "client-stage",
            "flops": float(macs) * 2.0,
            "macs": float(macs),
            "params": float(params),
        }
    except Exception:
        pass

    # 3) 保底：仅 patch_embed 的 Conv3d 理论计算（会低估完整 client 阶段）
    # patch_embed 内部会 reshape 为 [-1, C, T, P, P]，Conv3d(kernel=stride) 每个样本只产生 1 个空间位置。
    in_channels = patch_embed.in_channels
    out_channels = patch_embed.embed_dim
    k_t = patch_embed.temporal_patch_size
    k_h = patch_embed.patch_size
    k_w = patch_embed.patch_size
    n_samples = int(pixel_values.shape[0])
    macs = float(n_samples * out_channels * in_channels * k_t * k_h * k_w)
    return {
        "tool": "manual-conv3d",
        "scope": "patch-embed-only",
        "flops": macs * 2.0,
        "macs": macs,
        "params": float(sum(p.numel() for p in patch_embed.parameters())),
    }


def _count_client_active_params(visual_model) -> tuple[float, float]:
    total_params = float(sum(p.numel() for p in visual_model.parameters()))

    split_layer = int(getattr(visual_model, "client_layers", 1))
    split_layer = min(max(split_layer, 0), len(visual_model.blocks))

    active_modules = [visual_model.patch_embed, visual_model.pos_embed]
    active_modules.extend(list(visual_model.blocks[:split_layer]))

    # client 会在命中的 deepstack 层上执行 merger，因此把对应模块计入 active params。
    ds_indexes = set(getattr(visual_model, "deepstack_visual_indexes", []))
    for idx in range(split_layer):
        if idx in ds_indexes:
            ds_pos = visual_model.deepstack_visual_indexes.index(idx)
            active_modules.append(visual_model.deepstack_merger_list[ds_pos])

    active_params = float(sum(p.numel() for module in active_modules for p in module.parameters()))
    return total_params, active_params


def _sync_if_needed(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def main():
    parser = ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--packet", default=DEFAULT_PACKET_PATH)
    parser.add_argument("--profile-flops", default=True, help="统计 client 阶段（当前含前1层ViT）的计算量")
    parser.add_argument("--profile-latency", default=True, help="统计 client split 推理耗时（不含加载）")
    parser.add_argument("--latency-warmup", type=int, default=1, help="耗时统计前的预热次数")
    parser.add_argument("--latency-runs", type=int, default=1, help="耗时统计的正式运行次数")
    args = parser.parse_args()

    print("Loading client model...")
    # client 模式只跑到 patch_embed，输出可传输的视觉 token。
    model = Qwen3VLForSplitInference.from_pretrained(
        DEFAULT_CHECKPOINT,
        device_map="cpu" if args.cpu_only else "auto",
        trust_remote_code=True,
        mode="client",
    )
    processor = AutoProcessor.from_pretrained(DEFAULT_CHECKPOINT, trust_remote_code=True)
    device = model.device

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_TEXT}],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": DEFAULT_IMAGE_PATH}],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("processor 未返回 pixel_values 或 image_grid_thw")

    with torch.no_grad():
        print("[Client] Running vision split...")
        if args.profile_flops:
            stat = _profile_client_compute(model.model.visual, pixel_values, image_grid_thw)
            total_params, active_params = _count_client_active_params(model.model.visual)
            gflops = stat["flops"] / 1e9
            gmacs = stat["macs"] / 1e9
            print(
                "[Client] Compute Profile | "
                f"tool={stat['tool']} | "
                f"scope={stat.get('scope', 'unknown')} | "
                f"GFLOPs={gflops:.4f} | "
                f"GMACs={gmacs:.4f} | "
                f"Params(total)={total_params:.0f} | "
                f"Params(active)={active_params:.0f}"
            )
        if args.profile_latency:
            warmup = max(0, args.latency_warmup)
            runs = max(1, args.latency_runs)

            for _ in range(warmup):
                _ = model.model.visual(pixel_values, grid_thw=image_grid_thw)

            _sync_if_needed(device)
            start = time.perf_counter()
            for _ in range(runs):
                split_output = model.model.visual(pixel_values, grid_thw=image_grid_thw)
            _sync_if_needed(device)
            elapsed = time.perf_counter() - start

            avg_ms = elapsed * 1000.0 / runs
            total_ms = elapsed * 1000.0
            print(
                "[Client] Inference Latency | "
                f"runs={runs} | "
                f"avg={avg_ms:.3f} ms | "
                f"total={total_ms:.3f} ms"
            )
        else:
            split_output = model.model.visual(pixel_values, grid_thw=image_grid_thw)

    if isinstance(split_output, dict):
        vision_hidden_states = split_output["hidden_states"]
        split_layer = int(split_output.get("start_layer", 1))
        deepstack_prefix = split_output.get("deepstack_features", [])
    else:
        vision_hidden_states = split_output
        split_layer = 0
        deepstack_prefix = []

    packet = {
        "vision_hidden_states": vision_hidden_states.detach().cpu(),
        "split_layer": split_layer,
        "deepstack_prefix": [x.detach().cpu() for x in deepstack_prefix],
        "input_ids": _pick_tensor(inputs, "input_ids"),
        "attention_mask": _pick_tensor(inputs, "attention_mask"),
        "mm_token_type_ids": _pick_tensor(inputs, "mm_token_type_ids"),
        "image_grid_thw": _pick_tensor(inputs, "image_grid_thw"),
    }

    torch.save(packet, args.packet)
    print(f"[Client] Packet saved: {args.packet}")
    print("[Client] packet keys:", list(packet.keys()))
    print("Done.")


if __name__ == "__main__":
    main()