import torch
import numpy as np
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL3_5-1B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    use_flash_attn=False,
    trust_remote_code=True
).eval()

all_weights = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        all_weights.append(module.weight.data.clone().cpu().float().flatten().numpy())
all_weights = np.concatenate(all_weights)

del model
torch.cuda.empty_cache()

def per_tensor_quantize(tensor, bits):
    max_val = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / max_val
    scale = scale.clamp(min=1e-8)
    quantized = (tensor / scale).round().clamp(-max_val, max_val)
    return (quantized * scale)

def block_quantize(tensor, bits, block_size):
    numel = tensor.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    padded = torch.zeros(padded_numel, dtype=tensor.dtype, device=tensor.device)
    padded[:numel] = tensor.flatten()
    blocks = padded.reshape(-1, block_size)
    max_val = 2 ** (bits - 1) - 1
    scale = blocks.abs().amax(dim=1, keepdim=True) / max_val
    scale = scale.clamp(min=1e-8)
    quantized = (blocks / scale).round().clamp(-max_val, max_val)
    return (quantized * scale).flatten()[:numel]

def mx_quantize(tensor, bits, block_size):
    numel = tensor.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    padded = torch.zeros(padded_numel, dtype=tensor.dtype, device=tensor.device)
    padded[:numel] = tensor.flatten()
    blocks = padded.reshape(-1, block_size)
    max_val = 2 ** (bits - 1) - 1
    emax_elem = int(torch.floor(torch.log2(torch.tensor(float(max_val)))))
    block_max = blocks.abs().amax(dim=1, keepdim=True)
    shared_exp = torch.floor(torch.log2(block_max)) - emax_elem
    shared_exp = shared_exp.clamp(-127, 127)
    scale = (2.0 ** shared_exp).clamp(min=1e-8)
    quantized = (blocks / scale).round().clamp(-max_val, max_val)
    return (quantized * scale).flatten()[:numel]

chunk_size = 100_000_000
num_chunks = (len(all_weights) + chunk_size - 1) // chunk_size

def quantize_all(fn, **kwargs):
    result = []
    for i in range(num_chunks):
        s, e = i * chunk_size, min((i + 1) * chunk_size, len(all_weights))
        chunk = torch.from_numpy(all_weights[s:e]).to(torch.float32).cuda()
        result.append(fn(chunk, **kwargs).cpu().numpy())
        del chunk
        torch.cuda.empty_cache()
    return np.concatenate(result)

int4_weights = quantize_all(per_tensor_quantize, bits=4)
block_int4_weights = quantize_all(block_quantize, bits=4, block_size=32)
mxint4_weights = quantize_all(mx_quantize, bits=4, block_size=32)

def print_stats(name, weights):
    total = len(weights)
    w_gpu = torch.from_numpy(weights).cuda()
    abs_w = w_gpu.abs()

    print(f"\n{'=' * 80}")
    print(f"{name} WEIGHT DISTRIBUTION - STATISTICAL ANALYSIS")
    print(f"{'=' * 80}")

    print(f"\nTotal weights: {total:,}")
    print(f"Mean: {w_gpu.mean().item():.6f}")
    print(f"Std Dev: {w_gpu.std().item():.6f}")
    print(f"Min: {w_gpu.min().item():.6f}")
    print(f"Max: {w_gpu.max().item():.6f}")
    print(f"Median: {np.median(weights):.6f}")

    print(f"\nQ1 (25th): {np.percentile(weights, 25):.6f}")
    print(f"Q2 (50th): {np.percentile(weights, 50):.6f}")
    print(f"Q3 (75th): {np.percentile(weights, 75):.6f}")
    print(f"IQR: {np.percentile(weights, 75) - np.percentile(weights, 25):.6f}")

    positive = (w_gpu > 0).sum().item()
    negative = (w_gpu < 0).sum().item()
    zero = (w_gpu == 0).sum().item()
    print(f"\nPositive: {positive:,} ({positive/total*100:.2f}%)")
    print(f"Negative: {negative:,} ({negative/total*100:.2f}%)")
    print(f"Zero: {zero:,} ({zero/total*100:.2f}%)")

    print(f"\nMean |w|: {abs_w.mean().item():.6f}")
    print(f"Median |w|: {np.median(np.abs(weights)):.6f}")

    for t in [0.001, 0.01, 0.1]:
        count = (abs_w < t).sum().item()
        print(f"|w| < {t}: {count:>12,} weights ({count/total*100:>6.2f}%)")

    if name != "BF16":
        unique = torch.unique(w_gpu)
        print(f"\nUnique values: {len(unique):,}")

    del w_gpu, abs_w
    torch.cuda.empty_cache()

print_stats("BF16", all_weights)
print_stats("PER-TENSOR INT4", int4_weights)
print_stats("BLOCK-INT4", block_int4_weights)
print_stats("MXINT4", mxint4_weights)

block_int4_error = np.abs(all_weights - block_int4_weights)
mxint4_error = np.abs(all_weights - mxint4_weights)

print(f"\n{'=' * 80}")
print(f"QUANTIZATION ERROR COMPARISON")
print(f"{'=' * 80}")
print(f"\nBlock-INT4 mean error: {block_int4_error.mean():.6f}")
print(f"MXINT4 mean error:     {mxint4_error.mean():.6f}")
print(f"MXINT4 / Block-INT4:   {mxint4_error.mean() / block_int4_error.mean():.4f}x ({(mxint4_error.mean() / block_int4_error.mean() - 1) * 100:.1f}% higher)")

block_size = 32
max_val = 7
block_scales_list = []
mx_scales_list = []

for i in range(num_chunks):
    s, e = i * chunk_size, min((i + 1) * chunk_size, len(all_weights))
    chunk = torch.from_numpy(all_weights[s:e]).to(torch.float32).cuda()
    numel = chunk.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    padded = torch.zeros(padded_numel, dtype=chunk.dtype, device=chunk.device)
    padded[:numel] = chunk.flatten()
    blocks = padded.reshape(-1, block_size)

    b_scale = (blocks.abs().amax(dim=1) / max_val).clamp(min=1e-8)
    block_scales_list.append(b_scale.cpu().numpy())

    block_max = blocks.abs().amax(dim=1, keepdim=True)
    emax_elem = int(torch.floor(torch.log2(torch.tensor(float(max_val)))))
    shared_exp = torch.floor(torch.log2(block_max)) - emax_elem
    shared_exp = shared_exp.clamp(-127, 127)
    m_scale = (2.0 ** shared_exp.squeeze()).clamp(min=1e-8)
    mx_scales_list.append(m_scale.cpu().numpy())

    del chunk, padded, blocks, b_scale, block_max, shared_exp, m_scale
    torch.cuda.empty_cache()

block_scales = np.concatenate(block_scales_list)
mx_scales = np.concatenate(mx_scales_list)

print(f"\n{'=' * 80}")
print(f"SCALE GRANULARITY")
print(f"{'=' * 80}")
print(f"\nBlock-INT4 unique scales: {len(np.unique(block_scales)):,}")
print(f"MXINT4 unique scales:     {len(np.unique(mx_scales)):,}")
print(f"Reduction:                {(1 - len(np.unique(mx_scales)) / len(np.unique(block_scales))) * 100:.2f}%")