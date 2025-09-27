import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_prep import load_dataset_and_prepare, generate_ood_prompts
import os
from tqdm import tqdm
from experiment import get_activations_and_logits,run_patched
from causal_abstraction import CausalAbstraction
from analysis import analyze_results
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
hook_model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    tokenizer=tokenizer,
    device=device,
    dtype=torch.float16,
    center_writing_weights=False
)

# 加载 mib-bench/ioi 数据集，提取 prompt、choices 和 answerKey，设置最大序列长度。
id_prompts, choices, answer_keys, max_length = load_dataset_and_prepare(tokenizer, max_prompts=200)

# 定义多组实验条件（OOD 方案），创建结果目录。
ood_schemes = ['noise', 'language', 'adversarial', 'semantic', 'syntax']

results_dir = "exp1"
os.makedirs(results_dir, exist_ok=True)
all_results = []
ood_mediations = []

important_layers = [0,2,4,6,8,10,12,14,16,18,20,22]

hook_points=['attn.hook_q', 'attn.hook_k', 'attn.hook_v', 'attn.hook_z', 'mlp.hook_post', 'hook_resid_post']

id_logits,id_cache, id_attn_scores = get_activations_and_logits(hook_model, tokenizer, id_prompts, max_length=max_length,important_layers=important_layers)
# - id_logits: ID prompt 的模型输出，形状 [batch_size, seq_len, vocab_size]。
# - id_cache: ID prompt 的激活缓存，包含每层钩子点（如 attn.hook_q）。
# - id_attn_scores: ID prompt 的注意力分数。

id_patched_logits = {}
for layer in tqdm(important_layers):
    for h_p in hook_points:
        hook_point = f"blocks.{layer}.{h_p}"
        if hook_point not in id_patched_logits.keys():
            id_patched_logits[hook_point]=[]
        id_patched_logits[hook_point]=run_patched(hook_model, tokenizer, id_prompts, hook_point=hook_point, max_length=max_length)

causal_abs_id = CausalAbstraction(id_cache, layers=important_layers,hook_points=hook_points)
id_dag = causal_abs_id.build_dag(scheme="id")

id_faithfulness, _, id_diffs, _, _, id_accuracy, _, id_patched_accuracy, _ = analyze_results(id_logits, id_patched_logits, id_logits, id_patched_logits, choices,answer_keys, tokenizer, hook_points)

# 循环运行 OOD 方案（实验组），生成 OOD prompt，分析忠实度、准确率和中介效应。
all_results={}

for scheme in ood_schemes:
    print(f"\n运行 OOD 方案: {scheme}")
    ood_prompts, max_length = generate_ood_prompts(id_prompts, tokenizer, scheme=scheme)

    # 收集 OOD 数据激活。
    ood_logits, ood_cache, ood_attn_scores = get_activations_and_logits(hook_model, tokenizer, ood_prompts,max_length=max_length,important_layers=important_layers)

    ood_patched_logits = {}
    for layer in tqdm(important_layers):
        for h_p in hook_points:
            hook_point = f"blocks.{layer}.{h_p}"
            if hook_point not in ood_patched_logits.keys():
                ood_patched_logits[hook_point] = []
            ood_patched_logits[hook_point] = run_patched(hook_model, tokenizer, id_prompts, hook_point=hook_point,max_length=max_length)

    id_faithfulness, ood_faithfulness, id_diffs, ood_diffs, p_vals, id_accuracy, ood_accuracy, id_patched_accuracy, ood_patched_accuracy = analyze_results(
        id_logits, id_patched_logits, ood_logits, ood_patched_logits, choices,answer_keys, tokenizer, hook_points)

    all_results[scheme]={
        "Scheme": scheme,
        "ID Faithfulness": id_faithfulness,
        "OOD Faithfulness": ood_faithfulness,
        "ID diff": id_diffs,
        "ODD diff": ood_diffs,
        "p_vals":p_vals,
        "ID Accuracy": id_accuracy,
        "OOD Accuracy": ood_accuracy,
        "ID Patched Accuracy":id_patched_accuracy,
        "OOD Patched Accuracy": ood_patched_accuracy,
    }

#save as pkl
file = open('./%s/all_results.pickle'%results_dir, 'wb')
pickle.dump(all_results, file)
file.close()