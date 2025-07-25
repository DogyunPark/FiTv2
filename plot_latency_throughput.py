import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Example Data: [GFLOPs, FID]
methods = {
    "MDTv2-XL/2": [(259.1206*12, 25.0), (259.1206*24, 7), (259.1206*48, 2.77)],
    "SiT-XL/2": [(228.92*6, 12.91), (228.92*12, 4.75), (228.92*24, 2.53), (228.92*48, 1.95), (228.92*96, 1.84)],
    "REPA": [(228.92*6, 13.0), (228.92*12, 4.21), (228.92*24, 2.37), (228.92*48, 1.89), (228.92*96, 1.75)],
    "BFM-XL/2+SF (Ours)": [(215.6*6, 8.01), (215.6*12, 3.28), (215.6*24, 2.14), (215.6*48, 1.76), (215.6*96, 1.73)],
    "BFM-XL/2+SF+RA (Ours)": [(827.03, 3.31), (1277.24, 2.26), (1684.8, 2.14), (3782, 2.12)],
}

# Define a modern color palette
colors = ["#3498DB",    # Blue for MDTv2-XL/2
             "#9B59B6",    # Purple for SiT-XL/2
             "#2ECC71",    # Green for REPA
             "#E63946",    # Deep red for BFM-XL/2+SF (Ours)
             "#F77F00"]    # Orange for BFM-XL/2+SF+RA (Ours)
markers = ["o", "^", "s", "D", "P"]
line_styles = ["-", "-", "-", "-", "-"]

# Use clean matplotlib style without seaborn
plt.style.use('default')

# Set custom parameters for a clean, professional look
plt.rcParams.update({
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

# Customize the plot
#ax.set_xlabel("FLOPs (G)", fontweight='bold', fontsize=26)
ax.set_xlabel("FLOPs (G)", fontsize=26)
ax.set_ylabel("FID", fontsize=26)
#ax.set_title("Inference Latency vs. Image Quality Trade-off", fontweight='bold', pad=20)

# Set y-axis limits
ax.set_ylim(1, 15)  # Limit y-axis from 1.5 to 15

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