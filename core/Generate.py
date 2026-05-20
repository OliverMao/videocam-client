import torch
from argparse import ArgumentParser
from transformers import AutoProcessor
from Qwen3_5ModelSplit import Qwen3_5ForSplitInference

DEFAULT_IMAGE_PATH = '/home/nvidia/VLM/offline/image.png'
DEFAULT_TEXT = "图片中是否有违规行为，包括但不限于：1.吸烟 2.打架 3.着火 4.摔倒 5.赌博 6.其他违规行为。明确指出违规行为。输出如下：\n吸烟。\n如果图片中没有违规行为，输出如下：\n无。"
DEFAULT_CHECKPOINT = '/home/nvidia/Qwen3.5-VLM-Split/models'


def greedy_generate(
    model,
    input_ids,
    attention_mask=None,
    max_new_tokens=64,
    eos_token_id=None,
    **model_kwargs,
):
    generated_ids = input_ids

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
        **model_kwargs,
    )
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

        outputs = model(
            input_ids=next_token,
            attention_mask=torch.ones_like(next_token),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

    return generated_ids

def main():
    parser = ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()

    print("Loading model...")
    # 使用修改后的模型类加载
    model = Qwen3_5ForSplitInference.from_pretrained(
        DEFAULT_CHECKPOINT, 
        device_map='cpu' if args.cpu_only else 'auto',
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(DEFAULT_CHECKPOINT, trust_remote_code=True)
    device = model.device

    # 构造输入
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": DEFAULT_TEXT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": DEFAULT_IMAGE_PATH}
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        print("[Generate] Running autoregressive inference...")
        generation_inputs = {
            key: value
            for key, value in inputs.items()
            if value is not None
        }

        input_ids = generation_inputs.pop("input_ids")
        attention_mask = generation_inputs.pop("attention_mask", None)
        eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", eos_token_id)

        generated_ids = greedy_generate(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            **generation_inputs,
        )

        prompt_length = input_ids.shape[-1]
        generated_text = processor.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
        )[0].strip()

        print("[Generate] Output:")
        print(generated_text)

        
    print("\nDone.")

if __name__ == '__main__':
    main()