import matplotlib.pyplot as plt
import numpy as np

models = ['InternVL\n1B', 'Qwen3-VL\n2B', 'SmolVLM2\n2.2B', 'Gemma3\n4B',
          'Molmo\n7B', 'Eagle2.5\n8B', 'Phi-4\n14B']

block_int4_vqa = [68.17, 76.47, 73.43, 47.01, 56.29, 81.31, 67.64]
mxint4_vqa = [63.64, 73.23, 70.61, 39.43, 58.55, 77.78, 65.12]

block_int4_textvqa = [64.73, 75.50, 59.30, 58.52, 52.92, 81.25, 59.06]
mxint4_textvqa = [53.01, 71.88, 55.73, 50.63, 51.25, 77.84, 56.09]

vqa_penalty = [b - m for b, m in zip(block_int4_vqa, mxint4_vqa)]
textvqa_penalty = [b - m for b, m in zip(block_int4_textvqa, mxint4_textvqa)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

x_pos = np.arange(len(models))

ax1.plot(x_pos, block_int4_vqa, 'o', color='#2E86AB', markersize=12,
         linewidth=0, label='Block-INT4', markerfacecolor='#2E86AB', zorder=3)
ax1.plot(x_pos, mxint4_vqa, 's', color='#A23B72', markersize=10,
         linewidth=0, label='MXINT4', markerfacecolor='none', markeredgewidth=2.5, zorder=3)

for i in range(len(models)):
    ax1.plot([x_pos[i], x_pos[i]], [block_int4_vqa[i], mxint4_vqa[i]],
             '-', color='gray', linewidth=2, alpha=0.4, zorder=1)
    mid_y = (block_int4_vqa[i] + mxint4_vqa[i]) / 2
    if i == len(models) - 1:
        ax1.text(x_pos[i] - 0.15, mid_y, f'-{vqa_penalty[i]:.1f}%',
                 ha='right', va='center', fontsize=9, color='red', style='italic')
    else:
        ax1.text(x_pos[i] + 0.15, mid_y, f'-{vqa_penalty[i]:.1f}%',
                 ha='left', va='center', fontsize=9, color='red', style='italic')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylabel('VQA Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Model (sorted by size)', fontsize=13, fontweight='bold')
ax1.set_title('(a) VQAv2 Accuracy', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax1.set_ylim([35, 85])

ax2.plot(x_pos, block_int4_textvqa, 'o', color='#2E86AB', markersize=12,
         linewidth=0, label='Block-INT4', markerfacecolor='#2E86AB', zorder=3)
ax2.plot(x_pos, mxint4_textvqa, 's', color='#A23B72', markersize=10,
         linewidth=0, label='MXINT4', markerfacecolor='none', markeredgewidth=2.5, zorder=3)

for i in range(len(models)):
    ax2.plot([x_pos[i], x_pos[i]], [block_int4_textvqa[i], mxint4_textvqa[i]],
             '-', color='gray', linewidth=2, alpha=0.4, zorder=1)
    mid_y = (block_int4_textvqa[i] + mxint4_textvqa[i]) / 2
    if i == len(models) - 1:
        ax2.text(x_pos[i] - 0.15, mid_y, f'-{textvqa_penalty[i]:.1f}%',
                 ha='right', va='center', fontsize=9, color='red', style='italic')
    else:
        ax2.text(x_pos[i] + 0.15, mid_y, f'-{textvqa_penalty[i]:.1f}%',
                 ha='left', va='center', fontsize=9, color='red', style='italic')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=11)
ax2.set_ylabel('TextVQA Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Model (sorted by size)', fontsize=13, fontweight='bold')
ax2.set_title('(b) TextVQA Accuracy', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax2.set_ylim([45, 85])

plt.tight_layout()
plt.savefig('scatter_blockint4_vs_mxint4.png', dpi=300, bbox_inches='tight')
plt.show()