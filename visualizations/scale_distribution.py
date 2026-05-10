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

block_size = 32
bits = 4
max_val = 2 ** (bits - 1) - 1
chunk_size = 100_000_000
num_chunks = (len(all_weights) + chunk_size - 1) // chunk_size

block_int4_scales = []
mxint4_scales = []

for i in range(num_chunks):
    s, e = i * chunk_size, min((i + 1) * chunk_size, len(all_weights))
    chunk = torch.from_numpy(all_weights[s:e]).to(torch.float32).cuda()

    numel = chunk.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    padded = torch.zeros(padded_numel, dtype=chunk.dtype, device=chunk.device)
    padded[:numel] = chunk.flatten()
    blocks = padded.reshape(-1, block_size)

    block_scale = blocks.abs().amax(dim=1) / max_val
    block_scale = block_scale.clamp(min=1e-8)
    block_int4_scales.append(block_scale.cpu().numpy())

    block_max = blocks.abs().amax(dim=1, keepdim=True)
    emax_elem = int(torch.floor(torch.log2(torch.tensor(float(max_val)))))
    shared_exp = torch.floor(torch.log2(block_max)) - emax_elem
    shared_exp = shared_exp.clamp(-127, 127)
    mx_scale = (2.0 ** shared_exp.squeeze()).clamp(min=1e-8)
    mxint4_scales.append(mx_scale.cpu().numpy())

    del chunk, padded, blocks, block_scale, block_max, shared_exp, mx_scale
    torch.cuda.empty_cache()

block_int4_scales = np.concatenate(block_int4_scales)
mxint4_scales = np.concatenate(mxint4_scales)

bins = np.logspace(np.log10(1e-3), np.log10(1e-1), 200)

plt.figure(figsize=(12, 8))

plt.hist(block_int4_scales, bins=bins, color='#2ca02c', alpha=0.5,
         label=f'Block-INT4 ({len(np.unique(block_int4_scales)):,} unique scales)',
         edgecolor='black', linewidth=0.5)
plt.hist(mxint4_scales, bins=bins, color='#7030A0', alpha=0.5,
         label=f'MXINT4 ({len(np.unique(mxint4_scales)):,} unique scales)',
         edgecolor='black', linewidth=0.5)

plt.xscale('log')
plt.xlim(1e-3, 1e-1)
plt.xlabel('Scale Value (log scale)', fontsize=13, fontweight='bold')
plt.ylabel('Frequency', fontsize=13, fontweight='bold')
plt.title('Per-Block Scale Distribution Comparison', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--', which='both')
plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig('scale_distribution_overlay.png', dpi=300, bbox_inches='tight')
plt.show()