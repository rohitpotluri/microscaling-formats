import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

x_min, x_max = 2**-4, 2**3

x_fine = np.logspace(np.log2(x_min), np.log2(x_max), 2000, base=2)
y_block_fine = x_fine
y_mx_fine = 2.0 ** np.floor(np.log2(x_fine))

ax.fill_between(x_fine, y_mx_fine, y_block_fine,
                color='#d62728', alpha=0.12, zorder=1)

ax.plot([x_min, x_max], [x_min, x_max], color='#2ca02c',
        linewidth=3, label='Block-INT4', zorder=3)

pow_x = np.array([2**k for k in range(-4, 4)])
ax.plot(pow_x, pow_x, color='#7030A0', linewidth=3,
        drawstyle='steps-post', label='MXINT4', zorder=3)

worst_x = 2**0.92
worst_block = worst_x
worst_mx = 2**0
ax.plot(worst_x, worst_block, 'o', color='#2ca02c', markersize=11,
        zorder=5, markeredgecolor='black', markeredgewidth=1)
ax.plot(worst_x, worst_mx, 's', color='#7030A0', markersize=11,
        zorder=5, markeredgecolor='black', markeredgewidth=1)
ax.annotate('', xy=(worst_x, worst_mx), xytext=(worst_x, worst_block),
            arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2.2))
ax.text(worst_x * 1.18, np.sqrt(worst_block * worst_mx),
        r'up to 2$\times$ off',
        color='#d62728', fontweight='bold', fontsize=12, va='center')

avg_x = 2**-1.5
avg_block = avg_x
avg_mx = 2**-2
ax.plot(avg_x, avg_block, 'o', color='#2ca02c', markersize=11,
        zorder=5, markeredgecolor='black', markeredgewidth=1)
ax.plot(avg_x, avg_mx, 's', color='#7030A0', markersize=11,
        zorder=5, markeredgecolor='black', markeredgewidth=1)
ax.annotate('', xy=(avg_x, avg_mx), xytext=(avg_x, avg_block),
            arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=2.2))
ax.text(avg_x * 1.18, np.sqrt(avg_block * avg_mx),
        r'avg $\sqrt{2}\times$ off',
        color='#1f77b4', fontweight='bold', fontsize=12, va='center')

power_ticks = [2**k for k in range(-4, 4)]
power_labels = [f'$2^{{{k}}}$' for k in range(-4, 4)]
ax.set_xticks(power_ticks)
ax.set_xticklabels(power_labels, fontsize=11)
ax.set_yticks(power_ticks)
ax.set_yticklabels(power_labels, fontsize=11)

ax.set_xlabel('Optimal scale', fontsize=13, fontweight='bold')
ax.set_ylabel('Chosen scale', fontsize=13, fontweight='bold')

ax.grid(True, which='major', alpha=0.3, linestyle='--')

ax.set_xlim(x_min, x_max)
ax.set_ylim(x_min, x_max)
ax.set_aspect('equal')

ax.legend(fontsize=12, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig('mx_scale_geometry_2d.png', dpi=300, bbox_inches='tight')
plt.show()