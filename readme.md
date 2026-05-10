# Microscaling (MX) Quantization Benchmark Study

Benchmarking three INT quantization schemes — **per-tensor**, **block-32**, and **microscaling (MX)** — at three bit-widths (8, 4, 2) across **7 vision-language models** on VQAv2 and TextVQA validation sets.

📝 **Blog post with detailed findings:** [The Cost of Microscaling Formats](https://medium.com/@rohitpotluri1221/the-cost-of-microscaling-formats-8a0959ff54f6)

## Key Results

- **8-bit is essentially free.** Block-INT8 and MXINT8 stay within ±1% of BF16 baseline across all 7 models.
- **Per-tensor INT4 universally collapses.** Every model drops to ~0% accuracy — a phase transition, not gradual degradation. The collapse is mathematically forced: a single scale per tensor rounds ~90% of weights to exact zero.
- **Block-INT4 outperforms MXINT4** on TextVQA (7/7 models) and VQAv2 (5/7 models), with a 3–9% accuracy gap.
- **The MXINT4 cost is bounded.** The power-of-2 scale constraint (E8M0) forces a worst-case 2× and expected √2 ≈ 1.41× scale mismatch. Empirically, MXINT4 shows a 32% higher mean quantization error (1.32×), consistent with the theoretical bound.
- **2-bit is uniformly dead.** No scheme survives at 2 bits across any model.

## Models

| Model | Parameters | Linear Layers |
|---|---|---|
| InternVL3.5-1B | ~1B | 364 |
| Qwen3-VL-2B | ~2B | 295 |
| SmolVLM2-2.2B | ~2.2B | 401 |
| Gemma3-4B | ~4B | 295 |
| Molmo-7B-D | ~7B | 401 |
| Eagle2.5-8B | ~8B | 364 |
| Phi-4-multimodal | ~14B | 1005 |

## Quantization Schemes

| Scheme | Scale Type | Scale Granularity |
|---|---|---|
| Per-tensor INT | max(\|w\|) / max_val | 1 scale per tensor |
| Block-32 INT | block_max / max_val (BF16) | 1 scale per 32 weights |
| MXINTk | 2^(floor(log2(block_max)) - emax) (E8M0) | 1 power-of-2 scale per 32 weights |

Quantization is applied to `nn.Linear` weights only. Activations and embeddings remain in BF16.

## Results

### VQAv2_val Accuracy (%)

| Scheme | Eagle 8B | Gemma 4B | InternVL 1B | Qwen 2B | Molmo 7B | Phi-4 | SmolVLM 2.2B |
|---|---|---|---|---|---|---|---|
| **BF16** | **82.36** | **48.83** | **72.90** | **78.59** | **59.07** | **71.57** | **62.77** |
| INT8 | 81.05 | 44.91 | 72.08 | 78.37 | 58.97 | 70.33 | 54.63 |
| Block-INT8 | 82.41 | 48.49 | 72.79 | 78.27 | 59.00 | 71.57 | 63.17 |
| MXINT8 | 82.51 | 48.01 | 72.58 | 78.51 | 59.07 | 71.35 | 63.69 |
| INT4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.47 | 0.02 | 0.00 |
| Block-INT4 | 81.31 | 47.01 | 68.17 | 76.47 | 56.29 | 67.64 | 73.43 |
| MXINT4 | 77.78 | 39.43 | 63.64 | 73.23 | 58.55 | 65.12 | 70.61 |
| INT2 | 0.07 | 0.00 | 0.07 | 0.00 | 0.07 | 0.00 | 0.00 |
| Block-INT2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MXINT2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### TextVQA_val Accuracy (%)

| Scheme | Eagle 8B | Gemma 4B | InternVL 1B | Qwen 2B | Molmo 7B | Phi-4 | SmolVLM 2.2B |
|---|---|---|---|---|---|---|---|
| **BF16** | **82.65** | **59.19** | **68.30** | **79.84** | **56.63** | **63.03** | **64.50** |
| INT8 | 82.11 | 57.09 | 67.17 | 79.25 | 57.29 | 61.70 | 60.81 |
| Block-INT8 | 82.75 | 59.41 | 68.35 | 79.75 | 56.37 | 62.93 | 64.19 |
| MXINT8 | 82.59 | 59.76 | 59.95 | 80.34 | 56.67 | 63.15 | 64.50 |
| INT4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| Block-INT4 | 81.25 | 58.52 | 64.73 | 75.50 | 52.92 | 59.06 | 59.30 |
| MXINT4 | 77.84 | 50.63 | 53.01 | 71.88 | 51.25 | 56.09 | 55.73 |
| INT2 | 0.41 | 0.00 | 0.41 | 0.00 | 0.41 | 0.00 | 0.00 |
| Block-INT2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MXINT2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |

## Evaluation

- **Datasets:** 5,000-sample subsets of VQAv2 and TextVQA validation sets
- **Metric:** Official VQA accuracy — min(matches/3, 1) over 10 ground-truth answers per question
- **Baseline:** BF16 (no quantization)

## Setup

- PyTorch 2.5.1 + CUDA 12.1
- Transformers 4.55.4
- Flash Attention 2.7.2
- MX implementation follows the OCP Microscaling specification with E8M0 shared exponents

##

If you find this work useful, please consider starring the repo ⭐ and sharing the [blog post](https://medium.com/@rohitpotluri1221/the-cost-of-microscaling-formats-8a0959ff54f6).