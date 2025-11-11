# Stress-Testing IOI Circuits Under Distribution Shift: Hook-by-Layer Robustness in Qwen2.5-0.5B

This repository contains the official implementation for the paper: **"Stress-Testing IOI Circuits Under Distribution Shift: Hook-by-Layer Robustness in Qwen2.5-0.5B"**.

## Abstract

Circuit analyses of language models often rest on in-distribution (ID) case studies, leaving open whether the same internal pathways support behavior under distribution shift. We probe this question on *Indirect Object Identification* (IOI) using **Qwen2.5-0.5B**. We generate five controlled out-of-distribution (OOD) prompt families (noise, language, adversarial, semantic, syntax) while preserving IOI labels, and we perform zero-patch ablations across a fixed grid of 12 layers and 6 hook sites (`attn.hook_q,k,v,z`, `mlp.hook_post`, `hook_resid_post`). This study reveals that IOI behavior in Qwen2.5-0.5B is not carried by a single brittle route; instead, OOD perturbations re-weight which layers and hook families are critical for the task.

## Key Findings

1.  **Attention Output/Values as Sentinels**: Attention outputs (`attn.hook_z`) and values (`attn.hook_v`) in early layers are primary indicators of circuit failure. They exhibit the strongest sensitivity to ablations in-distribution and the largest performance degradation under OOD shifts.

2.  **OOD Fingerprints**: Different types of distribution shifts create distinct "fingerprints" of circuit disruption.
    *   **Language and Syntax** edits induce the broadest, most significant layer-wise changes.
    *   **Adversarial and Noise** shifts result in more localized, specific patterns of failure.

3.  **Pathway Reweighting, Not Uniform Failure**: The IOI circuit is not a single, brittle pathway. OOD conditions cause the model to re-weight the importance of different layers and components. We introduce a signed mediation statistic that reveals where attention's contribution is counteracted by the residual stream, signaling pathway failure and compensation.

4.  **Methodological Recommendations**: For robust circuit analysis, we recommend reporting hook-by-layer sensitivity maps, per-example distributions (not just averages), and signed mediation profiles to make claims that generalize beyond the in-distribution data.

## Repository Structure

```
.
├── exp.py                  # Main script to run all experiments
├── data_prep.py            # Handles dataset loading and OOD prompt generation
├── experiment.py           # Core logic for running patched forward passes on the model
├── analysis.py             # Computes metrics: faithfulness (KL), accuracy, t-tests
├── causal_abstraction.py   # Implements mediation analysis and causal DAGs
├── visual_result.py        # Generates plots from saved experimental results
├── exp1/                     # Default directory for results (plots, pickles)
└── mib_data/               # Default directory for cached Hugging Face datasets
```

## Setup and Installation

### 1. Clone the repository:
```bash
git clone https://github.com/williamli-15/ioi_ood.git
cd ioi_ood
```

### 2. Create a Conda environment and install dependencies:
It is recommended to use a Conda environment to manage dependencies. This code requires PyTorch with CUDA support.

```bash
conda create -n ioi-ood python=3.11
conda activate ioi-ood

# Install PyTorch (ensure it matches your CUDA version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

**`requirements.txt`:**
```
transformers
transformer_lens
datasets
scipy
numpy
spacy
matplotlib
seaborn
networkx
tqdm
```

### 3. Download the SpaCy model:
The syntactic OOD generation requires a SpaCy model.

```bash
python -m spacy download en_core_web_sm
```

## How to Reproduce Results

The entire experimental pipeline can be run with two main commands.

### 1. Run the Experiments
This script will:
1.  Download the `mib-bench/ioi` dataset.
2.  Generate the five OOD prompt sets.
3.  Run the baseline (unpatched) and zero-patch ablation experiments for both ID and all OOD sets.
4.  Perform the analysis (faithfulness, accuracy, KL divergence, p-values).
5.  Save all results to `exp1/all_results.pickle`.

Execute the main experiment script:
```bash
python exp.py
```
**Note**: This process is computationally intensive and may take a significant amount of time and VRAM, as it involves numerous forward passes of the Qwen2.5-0.5B model.

### 2. Generate Visualizations
After the experiments are complete, you can generate the plots from the paper (faithfulness and accuracy curves). This script reads the `exp1/all_results.pickle` file.

Execute the visualization script:
```bash
python visual_result.py
```
The resulting plots will be saved as PNG files in the `exp1/` directory, corresponding to the figures in the paper.

## Citation

This work has not been formally published. If you use this code or our findings in your research, please cite this repository.

```bibtex
@misc{li2025stress,
  author       = {Hengxu Li},
  title        = {Stress-Testing IOI Circuits Under Distribution Shift: Hook-by-Layer Robustness in Qwen2.5-0.5B},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/williamli-15/ioi_ood}}
}
```

## License
This project is licensed under the MIT License.