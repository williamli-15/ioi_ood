from scipy import stats
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import kl_div
import torch

def compute_faithfulness(original, patched, eps=1e-8):
    patched_log = F.log_softmax(patched,-1)
    original = F.softmax(original,-1)
    diff = kl_div(patched_log, original, reduction='batchmean').item()
    norm = original.norm().item()
    return 1 - (diff / (norm + eps))

def compute_kl(original, patched):
    patched_log = F.log_softmax(patched,-1)
    original = F.softmax(original,-1)
    kl = kl_div(patched_log, original, reduction='batchmean').item()
    return kl

def compute_accuracy(logits, choices, answer_keys, tokenizer):
    # 计算 IOI 任务准确率，基于 answerKey。
    correct = 0
    total = len(answer_keys)
    for i in range(total):
        pred_logits = logits[i, :]
        pred_probs = torch.softmax(pred_logits, dim=-1)
        choice_tokens = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices[i]]
        pred_idx = torch.argmax(pred_probs[choice_tokens]).item()
        if pred_idx == answer_keys[i]:
            correct += 1
    return correct / total if total > 0 else 0.0

def analyze_results(id_logits, id_patched_logits, ood_logits, ood_patched_logits,choices, answer_keys, tokenizer, hook_points=[0, 5, 12, 23]):
    # 功能：计算重要层忠实度、准确率、差异，执行 t 检验。
    id_faithfulness = [compute_faithfulness(id_logits, id_patched_logits[i]) for i in id_patched_logits.keys()]
    ood_faithfulness = [compute_faithfulness(ood_logits, ood_patched_logits[i]) for i in ood_patched_logits.keys()]

    id_diffs = [[compute_kl(id_logits[i], id_patched_logits[l][i]) for i in range(len(id_logits))] for l in id_patched_logits.keys()]
    ood_diffs = [[compute_kl(ood_logits[i], ood_patched_logits[l][i]) for i in range(len(id_logits))] for l in ood_patched_logits.keys()]
    p_vals = [stats.ttest_ind(id_diffs[l], ood_diffs[l])[1] for l in range(len(ood_patched_logits.keys()))]
    id_accuracy = compute_accuracy(id_logits, choices, answer_keys, tokenizer)
    ood_accuracy = compute_accuracy(ood_logits, choices, answer_keys, tokenizer)
    id_patched_accuracy = [compute_accuracy(id_patched_logits[i], choices, answer_keys, tokenizer) for i in id_patched_logits.keys()]
    ood_patched_accuracy = [compute_accuracy(ood_patched_logits[i], choices, answer_keys, tokenizer) for i in ood_patched_logits.keys()]
    return id_faithfulness, ood_faithfulness,id_diffs,ood_diffs,p_vals,id_accuracy,ood_accuracy,id_patched_accuracy,ood_patched_accuracy
