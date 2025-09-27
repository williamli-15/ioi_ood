import torch
from torch.nn.functional import kl_div, softmax
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 Qwen2.5-0.5B 模型和 tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_activations_and_logits(model, tokenizer, prompts, batch_size=16, max_length=32, important_layers=[0, 5, 12, 23]):
    # 收集重要层的激活（attn.hook_q, attn.hook_attn_scores, mlp.hook_post, hook_resid_pre）、logits 和注意力分数。
    all_logits, all_caches, all_attn_scores = [], [], {f"layer{l}": [] for l in important_layers}
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        tokens = inputs["input_ids"].to(model.cfg.device)
        attention_mask = inputs["attention_mask"].to(model.cfg.device)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, attention_mask=attention_mask)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Batch {i//batch_size} logits 包含 NaN 或 inf")
            continue
        all_logits.append(logits[:,-1].to('cpu'))
        all_caches.append(cache.to('cpu'))
        for l in important_layers:
            attn_scores = cache[f"blocks.{l}.attn.hook_attn_scores"]
            all_attn_scores[f"layer{l}"].append(attn_scores.to('cpu'))
    try:
        return (torch.cat(all_logits, dim=0),
                {k: torch.cat([c[k] for c in all_caches], dim=0) for k in all_caches[0]},
                {k: torch.cat(all_attn_scores[k], dim=0) for k in all_attn_scores})
    except RuntimeError as e:
        print(f"拼接失败: {e}")
        for i, logit in enumerate(all_logits):
            print(f"Batch {i} logits 形状: {logit.shape}")
        for l in important_layers:
            for i, attn in enumerate(all_attn_scores[f"layer{l}"]):
                print(f"Batch {i} layer{l} attn_scores 形状: {attn.shape}")
        raise

def zero_patch_hook(activation, hook):
    activation[:, :, :] = 0
    return activation

def run_patched(model, tokenizer, prompts, hook_point="blocks.5.attn.hook_q", batch_size=16, max_length=32):
    # 功能：运行 patching 实验，干预特定激活点。
    all_patched_logits = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        tokens = inputs["input_ids"].to(model.cfg.device)
        attention_mask = inputs["attention_mask"].to(model.cfg.device)
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, zero_patch_hook)],
                attention_mask=attention_mask
            )
        if torch.isnan(patched_logits).any() or torch.isinf(patched_logits).any():
            print(f"Patched Batch {i//batch_size} logits 包含 NaN 或 inf")
            continue
        all_patched_logits.append(patched_logits[:,-1].to('cpu'))
    return torch.cat(all_patched_logits, dim=0)