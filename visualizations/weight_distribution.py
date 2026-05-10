import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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

bins = np.linspace(-0.06, 0.06, 500)
bin_centers = (bins[:-1] + bins[1:]) / 2

datasets = {
    'BF16 (Original)': (all_weights, '#2E86AB', '-', 1.8, 0.00, 4.23),
    'Block-INT4 (BS=32)': (block_int4_weights, '#2ca02c', (0, (5, 2)), 1.8, 14.27, 14.27),
    'MXINT4 (BS=32)': (mxint4_weights, '#7030A0', '--', 1.8, 18.10, 18.10),
    'Per-Tensor INT4': (int4_weights, '#d62728', ':', 2.2, 90.61, 90.61),
}

plt.figure(figsize=(14, 9))

for name, (data, color, ls, lw, zeros, near_zero) in datasets.items():
    hist, _ = np.histogram(data, bins=bins, density=True)
    smoothed = gaussian_filter1d(hist, sigma=3)
    label = f"{name} ({zeros}% zeros, {near_zero}% |w|<0.001)"
    plt.plot(bin_centers, smoothed, color=color, linestyle=ls, linewidth=lw,
             alpha=0.9, label=label)
    plt.fill_between(bin_centers, smoothed, alpha=0.20, color=color)

plt.xlabel('Weight Value', fontsize=13, fontweight='bold')
plt.ylabel('Probability Density', fontsize=13, fontweight='bold')
plt.xlim([-0.06, 0.06])
plt.ylim([0, 90])
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('weight_distribution_final.png', dpi=300, bbox_inches='tight')
plt.show()