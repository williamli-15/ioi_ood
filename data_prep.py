from datasets import load_dataset, load_from_disk
import os
import random
from transformers import AutoTokenizer
import random
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_dataset_and_prepare(tokenizer, max_prompts=50):
    # 功能：加载 mib-bench/ioi 数据集，提取 prompt、choices 和 answerKey。
    os.makedirs("mib_data", exist_ok=True)
    try:
        ioi_dataset = load_from_disk("mib_data/ioi")
    except Exception as e:
        print(f"加载本地数据集失败: {e}. 尝试在线下载...")
        try:
            ioi_dataset = load_dataset("mib-bench/ioi", split="train")
            ioi_dataset.save_to_disk("mib_data/ioi")
        except Exception as e:
            print(f"在线下载失败: {e}. 请检查网络或手动下载数据集。")
            raise
    print("数据集列名:", ioi_dataset.column_names)
    print("样本示例:", ioi_dataset['train'][0])
    id_prompts = ioi_dataset['train']["prompt"][:max_prompts]
    choices = ioi_dataset['train']["choices"][:max_prompts]
    answer_keys = ioi_dataset['train']["answerKey"][:max_prompts]
    lengths = [len(tokenizer(p, add_special_tokens=True)["input_ids"]) for p in id_prompts]
    max_length = max(lengths) + 5
    print(f"ID Prompt 长度范围: {min(lengths)}-{max(lengths)}, 设置 max_length={max_length}")
    return id_prompts, choices, answer_keys, max_length

# 加载 spaCy 用于语法分析
nlp = spacy.load("en_core_web_sm")

translation_dict = {
    "to": "给",
    "at": "在",
    "while": "当",
    "gave": "给了",
    "offered": "提供了",
    "said": "说",
    "book": "书",
    "fridge": "冰箱",
    "duster": "掸子",
    "button": "纽扣",
    "picture frame": "相框",
    "plant": "植物",
    "rug": "地毯",
    "bottle of glue": "一瓶胶水",
    "sandpaper": "砂纸",
    "zipper": "拉链",
    "nail": "钉子",
    "watch": "手表",
    "walnut": "核桃",
    "pair of gloves": "一双手套",
    "bone": "骨头",
    "piece of chalk": "一支粉笔",
    "hairbrush": "发刷",
    "lamp": "灯",
    "computer": "电脑",
    "pair of pants": "一条裤子",
    "pitcher": "水壶",
    "rope": "绳子",
    "clipboard": "剪贴板",
    "doorknob": "门把手",
    "ladle": "勺子",
    "towel": "毛巾",
    "suitcase": "行李箱",
    "hammer": "锤子",
    "buckle": "扣子",
    "lightbulb": "灯泡",
    "colander": "漏勺",
    "belt": "腰带",
    "sofa": "沙发",
    "calendar": "日历",
    "keychain": "钥匙扣",
    "stapler": "订书机",
    "dryer": "吹风机",
    "thermometer": "温度计",
    "bed": "床",
    "marker": "记号笔",
    "calculator": "计算器",
    "piece of fabric": "一块布",
    "pot": "锅",
    "plunger": "柱塞",
    "cloth": "布",
    "shovel": "铲子",
    "purse": "钱包",
    "printer": "打印机",
    "bin": "垃圾桶",
    "drawer": "抽屉",
    "monitor": "显示器",
    "swab": "棉签",
    "pair of earrings": "一对耳环",
    "roll of tape": "一卷胶带",
    "blanket": "毯子",
    "table": "桌子",
    "keyboard": "键盘",
    "mouse": "鼠标",
    "tray": "托盘",
    "canvas": "画布",
    "whisk": "搅拌器",
    "bottle": "瓶子",
    "case": "箱子",
    "paper plane": "纸飞机",
    "lock": "锁",
    "hook": "钩子",
    "vase": "花瓶",
    "peeler": "削皮器",
    "bottle of lotion": "一瓶润肤露",
    "bolt": "螺栓",
    "pair of shorts": "一条短裤",
    "jug": "罐子",
    "saw": "锯子",
    "wardrobe": "衣柜",
    "picture": "图片",
    "grill": "烤架",
    "kettle": "水壶",
    "dustpan": "簸箕",
    "television": "电视",
    "hanger": "衣架",
    "piece of sandpaper": "一块砂纸",
    "pair of scissors": "一把剪刀",
    "broom": "扫帚",
    "tie": "领带",
    "roll": "卷",
    "piece": "块",
    "pair": "对",
    "bottle of lotion": "一瓶乳液",
    "piece of cloth": "一块布料",
    "piece of tape": "一节胶带",
    "piece of fabric": "一块织物",
    "hair brush": "发梳",
    "light bulb": "电灯泡"
}
def generate_ood_prompts(id_prompts, tokenizer, scheme='random'):
    """
    生成 OOD prompt，支持随机选择一种生成方案（噪声、语言变体、释义、对抗扰动、语义偏移、语法扰动）。
    参数：
    - id_prompts: 分布内 prompt 列表。
    - tokenizer: 用于计算 token 长度。
    - scheme: 'random'（随机选择一种方法）或具体方案名称（noise, language, paraphrase, adversarial, semantic, syntax）。
    创新点：随机化 OOD 生成增强实验灵活性，新增语法扰动模拟复杂语言变化，贴近真实世界场景。
    """
    ood_prompts = []
    schemes = ['noise', 'language', 'adversarial', 'semantic', 'syntax']
    selected_scheme = random.choice(schemes) if scheme == 'random' else scheme
    print(f"选择的 OOD 生成方案: {selected_scheme}")

    # 方案 1: 噪声添加（语义保留的拼写错误和动态噪声）
    if selected_scheme == 'noise':
        def typo_noise(prompt, noise_level=0.2):
            words = prompt.split()
            for i in range(len(words)):
                if random.random() < noise_level and len(words[i]) > 1:
                    pos = random.randint(0, len(words[i])-1)
                    words[i] = words[i][:pos] + random.choice("qwertyuiopasdfghjklzxcvbnm") + words[i][pos+1:]
            return " ".join(words)
        ood_prompts = [typo_noise(p, noise_level=random.uniform(0.1, 0.3)) for p in id_prompts]  # 动态噪声水平

    # 方案 2: 语言变体（使用 T5-small 模型进行完整多语言翻译，支持多种目标语言）
    elif selected_scheme == 'language':
        def translate_prompt(prompt, num_words=4):
            doc = nlp(prompt)
            # 提取非标点、非人名的单词
            words = [token.text for token in doc if not token.is_punct and not token.ent_type_ == "PERSON"]
            # 过滤出在翻译字典中的单词
            translatable_words = [w for w in words if w.lower() in translation_dict]
            # 随机选择 num_words 个单词
            num_words = min(num_words, len(translatable_words))
            if num_words == 0:
                return prompt
            words_to_translate = random.sample(translatable_words, num_words)
            # 复制原始单词列表
            result_words = [token.text for token in doc]
            # 翻译选中的单词
            for word in words_to_translate:
                translated_word = translation_dict.get(word.lower(), word)
                for i, w in enumerate(result_words):
                    if w.lower() == word.lower():
                        result_words[i] = translated_word
            return " ".join(result_words)
        ood_prompts = [translate_prompt(p) for p in id_prompts]

    # 对抗扰动（词序交换 + 语义混淆，模拟真实对抗）
    elif selected_scheme == 'adversarial':
        def adversarial_perturb(prompt):
            words = prompt.split()
            if len(words) > 2:
                swap_indices = random.sample(range(len(words)), 2)
                words[swap_indices[0]], words[swap_indices[1]] = words[swap_indices[1]], words[swap_indices[0]]
            return " ".join(words) + random.choice(["??", "!", "..."])  # 动态混淆后缀
        ood_prompts = [adversarial_perturb(p) for p in id_prompts]

    # 语义偏移（改进：多层次逻辑变换，如否定、条件、反事实）
    elif selected_scheme == 'semantic':
        def complex_semantic_shift(prompt):
            shifts = [
                f"If {prompt}, then it might not happen.",
                f"Although {prompt}, the opposite could be true.",
                f"Not only {prompt}, but also an alternative.",
                f"Suppose {prompt} didn't occur."
            ]
            return random.choice(shifts)
        ood_prompts = [complex_semantic_shift(p) for p in id_prompts]

    # 语法扰动（使用 spaCy 解析句法树，进行自然从句插入或语序调整）
    elif selected_scheme == 'syntax':
        def syntax_transform(prompt):
            doc = nlp(prompt)
            sentences = list(doc.sents)
            if len(sentences) > 1:
                # 反转句子顺序
                return " ".join([str(sentences[i]) for i in reversed(range(len(sentences)))])
            else:
                # 插入从句或调整语序
                words = [token.text for token in doc]
                if len(words) > 3:
                    insert_pos = len(words) // 2
                    words.insert(insert_pos, random.choice(["while waiting", "although possible", "if needed"]))
                return " ".join(words)
        ood_prompts = [syntax_transform(p) for p in id_prompts]

    # 清理 prompt 并检查 token 长度，确保 OOD 与 ID 的差异
    ood_prompts = [p.replace("\n", " ").strip() for p in ood_prompts]
    ood_lengths = [len(tokenizer(p, add_special_tokens=True)["input_ids"]) for p in ood_prompts]
    max_length = max(ood_lengths) + 5
    print(f"OOD Prompt 长度范围: {min(ood_lengths)}-{max(ood_lengths)}, 最终 max_length={max_length}")
    return ood_prompts, max_length

if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ood_schemes = ['noise', 'language', 'adversarial', 'semantic', 'syntax']
    id_prompts, choices, answer_keys, max_length = load_dataset_and_prepare(tokenizer, max_prompts=1000)
    ood_prompts, _ = generate_ood_prompts(id_prompts, tokenizer, scheme='syntax')
    print(ood_prompts)