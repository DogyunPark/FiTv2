import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Example Data: [GFLOPs, FID]
methods = {
    "DiT-XL/2": [(0.38*12, 39.3), (0.38*24, 16.45), (0.38*36, 12.4), (0.38*192, 10.2)],
    "FiTv2-XL/2": [(0.31*12, 11.85), (0.31*24, 9.76), (0.31*36, 9.57), (0.31*192, 9.14)],
    "SiT-XL/2": [(0.38*12, 11.4), (0.38*24, 9.47), (0.38*36, 9.27), (0.38*192, 8.97)],
    #"BiT-XL/2 (Ours)": [(2.45*1, 13.7), (0.5608*1000, 10.59), (0.8287*1000, 9.6), (1.0830*1000, 9.16), (1.2682*1000, 8.88), (1.5020*1000, 8.72)],
}

# Define a modern color palette
colors = ["#FF5A5F", "#3498DB", "#2ECC71"]  # Vibrant red, blue, green
markers = ["o", "^", "s"]
line_styles = ["-", "--", "-."]

# Use clean matplotlib style without seaborn
plt.style.use('default')

# Set custom parameters for a clean, professional look
plt.rcParams.update({
    #'font.family': 'sans-serif',
    #'font.sans-serif': ['Arial'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    #'axes.edgecolor': 'gray',
    'axes.linewidth': 0.5,
    'figure.facecolor': 'white',
    #'axes.facecolor': '#F8F9FA'
})

# Create figure with custom layout
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])

# Plot Each Method
for i, (method, points) in enumerate(methods.items()):
    # Convert points to numpy arrays for easier handling
    points = np.array(points)
    latencies = points[:, 0]
    fids = points[:, 1]
    
    # Plot with log scale on y-axis
    ax.semilogy(latencies, fids, color=colors[i], linestyle=line_styles[i], 
               linewidth=2.5, alpha=0.7)
    #ax.plot(latencies, fids, color=colors[i], linestyle=line_styles[i], 
    #        linewidth=2.5, alpha=0.7)
    ax.scatter(latencies, fids, color=colors[i], marker=markers[i], s=180, 
               label=method, edgecolor='white', linewidth=1.5, zorder=10)
    
    # Add labels for each point with better positioning
    # for j, (lat, fid) in enumerate(zip(latencies, fids)):
    #     # Alternate label positions to avoid overlap
    #     if j % 2 == 0:
    #         y_offset = 1.1
    #     else:
    #         y_offset = 0.9
    #     ax.annotate(f"{fid:.2f}", 
    #                xy=(lat, fid), 
    #                xytext=(5, 0),
    #                textcoords="offset points",
    #                fontsize=10, 
    #                fontweight='bold',
    #                color=colors[i],
    #                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8))
    for lat, fid in zip(latencies, fids):
        plt.text(lat+1, fid, f"{fid:.2f}", fontsize=8, ha="left")

# Customize the plot
ax.set_xlabel("Latency (ms per image)", fontweight='bold')
ax.set_ylabel("FID ↓ (Lower is Better)", fontweight='bold')
ax.set_title("Inference Latency vs. Image Quality Trade-off", fontweight='bold', pad=20)

# Add a descriptive subtitle
#plt.figtext(0.5, 0.01, "Lower FID scores indicate better image quality. Points show different NFE settings.", 
           #ha="center", fontsize=12, fontstyle='italic')

# Customize y-axis to show non-scientific notation
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# Add a legend with custom styling
legend = ax.legend(title="Models", frameon=True, fancybox=True, framealpha=0.95, 
                  loc='upper right', edgecolor='gray')
legend.get_title().set_fontweight('bold')

# Add a border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('gray')
    spine.set_linewidth(0.5)

# Adjust layout and save with high DPI
plt.tight_layout()
plt.savefig("/hub_data2/dogyun/LWD/fitv2_lwd_cifar_latency_fid_2.png", dpi=300, bbox_inches='tight')