import torch
import numpy as np
import matplotlib.pyplot as plt
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

block_int4_weights = quantize_all(block_quantize, bits=4, block_size=32)
mxint4_weights = quantize_all(mx_quantize, bits=4, block_size=32)

block_int4_error = np.abs(all_weights - block_int4_weights)
mxint4_error = np.abs(all_weights - mxint4_weights)

bins = np.linspace(0, 0.01, 200)

plt.figure(figsize=(12, 8))

plt.hist(block_int4_error, bins=bins, color='#2ca02c', alpha=0.5,
         label=f'Block-INT4 (mean: {block_int4_error.mean():.6f})',
         edgecolor='black', linewidth=0.5)
plt.hist(mxint4_error, bins=bins, color='#7030A0', alpha=0.5,
         label=f'MXINT4 (mean: {mxint4_error.mean():.6f})',
         edgecolor='black', linewidth=0.5)

plt.axvline(block_int4_error.mean(), color='#2ca02c', linestyle='--', linewidth=2)
plt.axvline(mxint4_error.mean(), color='#7030A0', linestyle='--', linewidth=2)

plt.xlabel('Absolute Error |w_original - w_quantized|', fontsize=13, fontweight='bold')
plt.ylabel('Frequency', fontsize=13, fontweight='bold')
plt.title('Quantization Error Comparison', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig('quantization_error_overlay.png', dpi=300, bbox_inches='tight')
plt.show()