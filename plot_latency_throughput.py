import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Example Data: [GFLOPs, FID]
methods = {
    #"DiT-XL/2": [(0.38*12, 39.3), (0.38*24, 16.45), (0.38*36, 12.4), (0.38*192, 10.2)],
    #"DiT-XL/2": [(0.38*24, 16.45), (0.38*36, 12.4), (0.38*192, 10.2)],
    #"FiTv2-XL/2": [(0.31*12, 11.85), (0.31*24, 9.76), (0.31*36, 9.57), (0.31*192, 9.14)],
    #"SiT-XL/2": [(0.38*12, 11.4), (0.38*24, 9.47), (0.38*36, 9.27), (0.38*192, 8.97)],
    #"MDTv2-XL/2": [(17.09, 14.09), (51.02, 2.92), (102.34, 2.12)],
    #"FiTv2-XL/2": [(8.23, 14.9), (22.51, 3.64), (69.77, 2.33), (140.07, 2.21)],
    #"SiT-XL/2": [(6.54, 13.1), (13.78, 3.19), (43.3, 2.24), (87.7, 2.13)],
    #"REPA": [(14.28, 2.95), (45.3, 2.01), (91.9, 1.89)],
    #"MDTv2-XL/2": [(17.09, 14.09), (51.02, 2.92), (102.34, 2.12)],
    #"FiTv2-XL/2": [(8.23, 14.9), (22.51, 3.64), (69.77, 2.33), (140.07, 2.21)],
    "MDTv2-XL/2": [(259.1206*12, 25.0), (259.1206*24, 7), (259.1206*48, 2.77)],
    "SiT-XL/2": [(228.92*6, 15.24), (228.92*12, 4.26), (228.92*24, 2.45), (228.92*48, 1.89)],
    "REPA": [(237.3*6, 14.9), (237.3*12, 4.09), (237.3*24, 2.3)],
    "BFM-XL/2 (Ours)": [(215.6*6, 9.61), (215.6*12, 4.01), (215.6*24, 2.13), (215.6*48, 1.77)],
    #"BiT-XL/2 (Ours)": [(5.33, 13.48), (8.89, 11.74), (14.3, 11.46)],
}

# Define a modern color palette
colors = ["#FF5A5F", "#3498DB", "#2ECC71", "#9B59B6", "#F1C40F"]  # Vibrant red, blue, green
markers = ["o", "^", "s", "D", "P"]
line_styles = ["-", "-", "-", "-", "-"]

# Use clean matplotlib style without seaborn
plt.style.use('default')

# Set custom parameters for a clean, professional look
plt.rcParams.update({
    #'font.family': 'sans-serif',
    #'font.sans-serif': ['Arial'],
    'font.size': 24,
    'axes.labelsize': 20,
    'axes.titlesize': 30,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 24,
    'figure.titlesize': 30,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    #'axes.edgecolor': 'gray',
    'axes.linewidth': 2,
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
    #ax.semilogy(latencies, fids, color=colors[i], linestyle=line_styles[i], 
    #           linewidth=3, alpha=0.7)
    ax.plot(latencies, fids, color=colors[i], linestyle=line_styles[i], 
            linewidth=4, alpha=0.7)
    ax.scatter(latencies, fids, color=colors[i], marker=markers[i], s=100, 
               label=method, zorder=10)
    
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
    #for lat, fid in zip(latencies, fids):
    #    plt.text(lat+1, fid, f"{fid:.2f}", fontsize=8, ha="left")

# Customize the plot
#ax.set_xlabel("FLOPs (G)", fontweight='bold', fontsize=26)
ax.set_xlabel("FLOPs (G)", fontsize=26)
ax.set_ylabel("FID", fontsize=26)
#ax.set_title("Inference Latency vs. Image Quality Trade-off", fontweight='bold', pad=20)

# Set y-axis limits
#ax.set_ylim(1, 7)  # Limit y-axis from 1.5 to 15
#ax.set_xlim(1000, 8000)  # Limit y-axis from 1.5 to 15

# Add a descriptive subtitle
#plt.figtext(0.5, 0.01, "Lower FID scores indicate better image quality. Points show different NFE settings.", 
           #ha="center", fontsize=12, fontstyle='italic')

# Customize y-axis to show non-scientific notation
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# Add a legend with custom styling
legend = ax.legend(frameon=True, fancybox=True, framealpha=0.95, 
                  loc='upper right', edgecolor='gray')
legend.get_title().set_fontweight('bold')

# Add a border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('gray')
    spine.set_linewidth(0.5)

# Adjust layout and save with high DPI
plt.tight_layout()
# Save as PDF for maximum quality
plt.savefig("/hub_data2/dogyun/LWD/fitv2_lwd_cifar_latency_fid2.pdf", format="pdf", bbox_inches='tight')
## Also save as PNG for quick viewing
plt.savefig("/hub_data2/dogyun/LWD/fitv2_lwd_cifar_latency_fid.png", dpi=300, bbox_inches='tight')