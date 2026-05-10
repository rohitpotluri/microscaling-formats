import matplotlib.pyplot as plt

models = ['Eagle\n8B', 'Gemma\n4B', 'InternVL\n1B', 'Qwen\n2B', 'Molmo\n7B', 'Phi-4', 'SmolVLM\n2.2B']
schemes = ['BF16', 'INT8', 'Block-INT8', 'MXINT8',
           'INT4', 'Block-INT4', 'MXINT4',
           'INT2', 'Block-INT2', 'MXINT2']

vqa2 = [
    [82.36, 48.83, 72.90, 78.59, 59.07, 71.57, 62.77],
    [81.05, 44.91, 72.08, 78.37, 58.97, 70.33, 54.63],
    [82.41, 48.49, 72.79, 78.27, 59.00, 71.57, 63.17],
    [82.51, 48.01, 72.58, 78.51, 59.07, 71.35, 63.69],
    [0.00,  0.00,  0.00,  0.00,  0.47,  0.02,  0.00],
    [81.31, 47.01, 68.17, 76.47, 56.29, 67.64, 73.43],
    [77.78, 39.43, 63.64, 73.23, 58.55, 65.12, 70.61],
    [0.07,  0.00,  0.07,  0.00,  0.07,  0.00,  0.00],
    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
]

textvqa = [
    [82.65, 59.19, 68.30, 79.84, 56.63, 63.03, 64.50],
    [82.11, 57.09, 67.17, 79.25, 57.29, 61.70, 60.81],
    [82.75, 59.41, 68.35, 79.75, 56.37, 62.93, 64.19],
    [82.59, 59.76, 59.95, 80.34, 56.67, 63.15, 64.50],
    [0.00,  0.00,  0.00,  0.00,  0.05,  0.00,  0.00],
    [81.25, 58.52, 64.73, 75.50, 52.92, 59.06, 59.30],
    [77.84, 50.63, 53.01, 71.88, 51.25, 56.09, 55.73],
    [0.41,  0.00,  0.41,  0.00,  0.41,  0.00,  0.00],
    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
    [0.00,  0.00,  0.00,  0.00,  0.00,  0.03,  0.00],
]

def make_table(data, title, filename):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')

    cell_text = [[f'{v:.2f}' for v in row] for row in data]
    cell_colors = [['white'] * len(models) for _ in range(len(schemes))]

    table = ax.table(
        cellText=cell_text,
        rowLabels=schemes,
        colLabels=models,
        cellColours=cell_colors,
        rowColours=['#e8e8e8'] * len(schemes),
        colColours=['#e8e8e8'] * len(models),
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for j in range(len(models)):
        table[1, j].set_text_props(fontweight='bold')
    table[1, -1].set_text_props(fontweight='bold')

    for j in range(len(models)):
        table[0, j].set_text_props(fontweight='bold')

    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

make_table(vqa2, 'VQAv2$_{val}$ Accuracy (%)', 'table_vqa2val.png')
make_table(textvqa, 'TextVQA$_{val}$ Accuracy (%)', 'table_textvqa_val.png')