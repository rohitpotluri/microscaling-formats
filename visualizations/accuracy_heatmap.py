import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

models = ['InternVL\n1B', 'Qwen3-VL\n2B', 'SmolVLM2\n2.2B', 'Gemma3\n4B',
          'Molmo\n7B', 'Eagle2.5\n8B', 'Phi-4\n14B']

vqa_degradation = {
    'INT8': [-0.82, -0.22, -8.14, -3.92, -0.10, -1.31, -1.24],
    'Block-INT8': [-0.11, -0.32, 0.40, -0.34, -0.07, 0.05, 0.00],
    'MXINT8': [-0.32, -0.08, 0.92, -0.82, 0.00, 0.15, -0.22],
    'INT4 (COLLAPSE)': [-72.90, -78.59, -62.77, -48.83, -58.60, -82.36, -71.55],
    'Block-INT4': [-4.73, -2.12, 10.66, -1.82, -2.78, -1.05, -3.93],
    'MXINT4': [-9.26, -5.36, 7.84, -9.40, -0.52, -4.58, -6.45],
}

textvqa_degradation = {
    'INT8': [-1.13, -0.59, -3.69, -2.10, 0.66, -0.54, -1.33],
    'Block-INT8': [0.05, -0.09, -0.31, 0.22, -0.26, 0.10, -0.10],
    'MXINT8': [-8.35, 0.50, 0.00, 0.57, 0.04, -0.06, 0.12],
    'INT4 (COLLAPSE)': [-68.30, -79.84, -64.50, -59.19, -56.58, -82.65, -63.03],
    'Block-INT4': [-3.57, -4.34, -5.20, -0.67, -3.71, -1.40, -3.97],
    'MXINT4': [-15.29, -7.96, -8.77, -8.56, -5.38, -4.81, -6.94],
}

schemes = list(vqa_degradation.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

vqa_matrix = np.array([vqa_degradation[s] for s in schemes])
textvqa_matrix = np.array([textvqa_degradation[s] for s in schemes])

COLLAPSE_COLOR = '#4A4A4A'
COLLAPSE_TEXT_COLOR = 'white'
VMIN, VMAX = -16, 16
collapse_row_idx = 3

sns.heatmap(vqa_matrix, annot=False, cmap='RdBu', center=0,
            xticklabels=models, yticklabels=schemes,
            cbar_kws={'label': 'Accuracy Drop (%)'},
            linewidths=0.5, linecolor='#dddddd', ax=ax1,
            vmin=VMIN, vmax=VMAX, cbar=True, square=True)

for i, val in enumerate(vqa_matrix[collapse_row_idx]):
    rect = mpatches.Rectangle((i, collapse_row_idx), 1, 1,
                               linewidth=0.5, edgecolor='#dddddd',
                               facecolor=COLLAPSE_COLOR, zorder=10)
    ax1.add_patch(rect)
    ax1.text(i + 0.5, collapse_row_idx + 0.5, f'{val:.1f}',
             ha='center', va='center', color=COLLAPSE_TEXT_COLOR, fontsize=10,
             fontweight='bold', zorder=11)

for row_idx in range(len(schemes)):
    if row_idx != collapse_row_idx:
        for col_idx in range(len(models)):
            val = vqa_matrix[row_idx, col_idx]
            ax1.text(col_idx + 0.5, row_idx + 0.5, f'{val:.1f}',
                     ha='center', va='center', fontsize=10)

ax1.set_title('(a) VQAv2 Degradation Heatmap', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Model (sorted by size)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quantization Scheme', fontsize=12, fontweight='bold')

sns.heatmap(textvqa_matrix, annot=False, cmap='RdBu', center=0,
            xticklabels=models, yticklabels=schemes,
            cbar_kws={'label': 'Accuracy Drop (%)'},
            linewidths=0.5, linecolor='#dddddd', ax=ax2,
            vmin=VMIN, vmax=VMAX, cbar=True, square=True)

for i, val in enumerate(textvqa_matrix[collapse_row_idx]):
    rect = mpatches.Rectangle((i, collapse_row_idx), 1, 1,
                               linewidth=0.5, edgecolor='#dddddd',
                               facecolor=COLLAPSE_COLOR, zorder=10)
    ax2.add_patch(rect)
    ax2.text(i + 0.5, collapse_row_idx + 0.5, f'{val:.1f}',
             ha='center', va='center', color=COLLAPSE_TEXT_COLOR, fontsize=10,
             fontweight='bold', zorder=11)

for row_idx in range(len(schemes)):
    if row_idx != collapse_row_idx:
        for col_idx in range(len(models)):
            val = textvqa_matrix[row_idx, col_idx]
            ax2.text(col_idx + 0.5, row_idx + 0.5, f'{val:.1f}',
                     ha='center', va='center', fontsize=10)

ax2.set_title('(b) TextVQA Degradation Heatmap', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Model (sorted by size)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Quantization Scheme', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('heatmap_degradation_full.png', dpi=300, bbox_inches='tight')
plt.show()