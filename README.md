# Polynomial mixing for efficient self-supervised speech encoders

## In brief
This repository contains the code for the paper **“Polynomial mixing for efficient self-supervised speech encoders”** (ICASSP 2026). TODO add link to paper
It introduces the **Polynomial Mixer (PoM)**, a drop-in replacement for multi-head self-attention (MHA) with **linear complexity** in the input sequence length, and evaluates PoM using BEST-RQ self-supervised speech representation learning framework on downstream ASR.
The code contained in this repository is thought as a plug-in into **SpeechBrain** library. 

## Getting started

### Requirements

Recap: 
- **PyTorch** (TODO : version)
- **SpeechBrain** (version **1.0.3** in the experiments)
- Optional for decoding: a **language model** (KenLM) TODO add link to kenLM repo.
- **ffmpeg** or equivalent (TODO: version)

TODO: add requirements file for the repo; add command line to install the requirements 


### Data

Download from TODO add URL

- **Pre-training:** LibriSpeech **960h** (English audiobooks)
- **Fine-tuning:** LibriSpeech **train-100 (clean)** subset
- **Evaluation:** LibriSpeech *test-clean* and *test-other* (with and without kenLM language model)

### The code 

TODO add tree with content 

### Example configs
TODO see `hparams` subfolder under `recipes`

### Example jobs
TODO see examples in `run` subfolder 

## The method

### Motivation
Transformer-based speech encoders rely on **multi-head attention**, whose **quadratic** memory and compute cost in sequence length constrains scalability. PoM replaces token mixing by a linear-complexity operator that summarizes the sequence into a shared state and broadcasts selected information back to each token.

### Contributions
- A speech-tailored **Polynomial Mixer (PoM)** token mixer with linear complexity, designed as a drop-in replacement for MHA.
- Integration of PoM into a **BEST-RQ** self-supervised learning framework for speech encoders.
- Experiments showing PoM achieves competitive WER vs. MHA and other linear-complexity alternatives, with improved runtime/memory behavior.

### How the Polynomial Mixer works (PoM)
PoM maps an input matrix $X \in \mathbb{R}^{d \times n}$ to an output in $\mathbb{R}^{d \times n}$ via a polynomial global representation $H(X)$ and a token-wise selector $S$:
```math
\mathrm{PoM}(X) = W_o \left[ \sigma(W_s X) \circ H(X)\mathbf{1}^{\top} \right],
```
where $\circ$ is the Hadamard product, $\sigma$ is a sigmoid, and $\mathbf{1}\in\mathbb{R}^{n\times 1}$ is a vector of ones.

The global representation is computed with a fixed-degree polynomial over projected views:
```math
H(X) =
\left[
h(W_1 X) \;\middle|\;
h(W_1 X)\circ h(W_2 X) \;\middle|\; \dots \;\middle|\;
\prod_{m=1}^k h(W_m X)
\right]\mathbf{1},
```
and the selector is
```math
S=\sigma(W_s X).
```

### Figures (from the paper)

**Principle of PoM**
![Principle of the Polynomial Mixer. The input sequence is projected through k polynomial branches, aggregated into a global representation H(X), and combined with a token-wise selector S. The output is obtained by projecting the selected state back to the input space.](./figs/pom_principle_small.png "pom_principle")

**Inference time and peak memory**
![Inference time and peak memory usage of BEST-RQ models (~95M params) with various token mixers. Input length is increased from 10 to 80 seconds. MHA requires significantly more time and VRAM as the input size increases in comparison with linear alternatives, including PoM.](./figs/monitoring.png "monitoring logs")

### Main results on LibriSpeech (WER)
Models are pretrained on LibriSpeech-960h and fine-tuned on the *train-100* subset. 
Confidence intervals are computed from 1000 bootstrap trials. 
Results marked with † are reported from [Whetten at al.](https://doi.org/10.1109/SLT61566.2024.10832323) comparative study (2024). 
Lower is better: **best MHA variant** is in bold, and the *best linear mixer* is in italic.


**Base models:** ~95M parameters

| Model | Clean | Clean + LM | Other | Other + LM |
|---|---|---|---|---|
| RelPosMHA | **7.96 (± 0.32)** | **4.89 (± 0.25)** | **17.61** (± 0.54) | 12.13 (± 0.44) |
| RoPE MHA | **8.06** (± 0.31) | **4.90** (± 0.26) | **17.53 (± 0.48)** | **11.98 (± 0.45)** |
| regular MHA | 8.59 (± 0.32) | 5.37 (± 0.25) | 19.44 (± 0.54) | 13.47 (± 0.46) |
| PoM base | 8.31 (± 0.31) | *5.42* (± 0.27) | *19.06* (± 0.53) | *13.62* (± 0.48) |
| SummaryMixing | 9.79 (± 0.34) | 5.93 (± 0.27) | 22.80 (± 0.60) | 15.84 (± 0.51) |
| Mamba† | *7.61* | *5.50* (± 0.28) | 19.97 | 15.37 |
| HyperConformer† | 8.22 | 5.77 (± 0.28) | 19.29 | 15.03 |
| FastFormer† | 9.32 | 6.82 (± 0.31) | 22.75 | 17.95 |

**Large models:** ~315M parameters

| Model | Clean | Clean + LM | Other | Other + LM |
|---|---|---|---|---|
| RelPosMHA | **4.92 (± 0.25)** | **3.49 (± 0.21)** | **10.78 (± 0.37)** | **8.09 (± 0.35)** |
| RoPE MHA | 5.13 (± 0.26) | 3.66 (± 0.21) | 10.99 (±0.33) | 8.45 (± 0.37) |
| PoM base | 6.28 (± 0.26) | *4.52* (± 0.23) | 14.86 (± 0.47) | *11.33* (± 0.43) |
| SummaryMixing | 7.35 (± 0.31) | 4.85 (± 0.25) | 17.60 (± 0.53) | 12.97 (± 0.49) |
| Mamba† | *5.59* | *4.48* (± 0.25) | 15.47 | 12.66 |
| HyperConformer† | 5.87 | *4.54* (± 0.32) | *13.13* | *10.78* |
| FastFormer† | 13.16 | 9.89 (± 0.34) | 31.91 | 26.75 |

## Citation
If you use this work, please cite the paper:

```bibtex
@inproceedings{feillet2026pom,
  title   = {Polynomial mixing for efficient self-supervised speech encoders},
  author  = {Eva Feillet and Ryan Whetten and David Picard and Alexandre Allauzen},
  booktitle={2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year    = {2026},
  organization={IEEE}
}