import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set style for a professional look
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'

# Method names
methods = [
    "FM",
    "BFM",
    "BFM + SF",
    "BFM + SF + RA"
]

# Latency values (in seconds)
latency = [20.38, 13.54, 18.24, 11.33]

# FID values (None for missing entries)
fid = [120.7, 124.8, 98.32, 100.2]

# Bar position setup
x = np.arange(len(methods))
width = 0.3  # Slightly narrower bars for cleaner look

# Custom colors with better contrast
latency_color = '#3777af'  # Deeper blue
fid_color = '#e85f2d'      # Richer orange

# Create figure and axis with improved dimensions and DPI
fig, ax1 = plt.subplots(figsize=(6, 6), dpi=120)

# Add a subtle background color
fig.patch.set_facecolor('#f8f9fa')
ax1.set_facecolor('#f8f9fa')

# Normalize all values relative to FM (which becomes 100%)
normalized_latency = [100] + [lat/latency[0] * 100 for lat in latency[1:]]
normalized_fid = [100] + [f/fid[0] * 100 for f in fid[1:]]

# Plot latency bars on left y-axis with enhanced styling
bar1 = ax1.bar(x - width/2, normalized_latency, width, label='Latency (%)', 
               color=latency_color, alpha=0.9, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Latency (% of FM baseline)', fontsize=12, fontweight='bold', color=latency_color)
ax1.set_ylim(0, max(normalized_latency) * 1.3)
ax1.tick_params(axis='y', labelcolor=latency_color, labelsize=10)
ax1.tick_params(axis='x', labelsize=11)

# Plot FID bars on right y-axis with enhanced styling
ax2 = ax1.twinx()
fid_clean = [f if f is not None else 0 for f in fid]  # Replace None with 0 for plotting
bar2 = ax2.bar(x + width/2, fid_clean, width, label='FID', 
               color=fid_color, alpha=0.9, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('FID Score', fontsize=12, fontweight='bold', color=fid_color)
ax2.set_ylim(0, max([f for f in fid if f is not None]) * 1.3)
ax2.tick_params(axis='y', labelcolor=fid_color, labelsize=10)

# Improve x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11, fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')  # Horizontal labels for better readability

# Add value labels on top of bars
for i, v in enumerate(normalized_latency):
    ax1.text(i - width/2, v + max(normalized_latency)*0.03, f'{v:.1f}%', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color=latency_color)

for i, v in enumerate(fid):
    if v is not None:
        ax2.text(i + width/2, v + max([f for f in fid if f is not None])*0.03, f'{v:.1f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color=fid_color)
    else:
        ax2.text(i + width/2, max([f for f in fid if f is not None])*0.05, 'n/a', 
                 ha='center', va='bottom', fontsize=10, color='gray', fontweight='bold')

# Add a legend with enhanced positioning and style
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
                    frameon=True, framealpha=0.95, edgecolor='lightgray', 
                    facecolor='white', fontsize=10)

# Add a descriptive title and subtitle
# fig.suptitle("Performance Comparison of Flow Matching Methods", 
#              fontsize=16, fontweight='bold', y=0.98)
# ax1.set_title("Comparing Latency and FID Scores Across Different Approaches", 
#               fontsize=12, pad=10, color='#555555')

# Add a border to separate the plot visually
for spine in ['top', 'right', 'bottom', 'left']:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_color('#dddddd')
    ax1.spines[spine].set_linewidth(1)

# Add a subtle grid for the latency axis only (left)
ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#aaaaaa')

# Add a descriptive annotation
# plt.figtext(0.12, 0.02, 
#             "Lower values are better for both metrics.\nBFM + SF + RA achieves the best latency performance.", 
#             fontsize=9, style='italic', color='#555555')

# # Add a subtle watermark-like credit
# plt.figtext(0.98, 0.02, "Data Visualization", 
#             ha='right', fontsize=8, color='#bbbbbb')

# Improve layout for better spacing
fig.tight_layout(pad=1.5)
plt.subplots_adjust(top=0.88)

# Save with higher quality
plt.savefig("/hub_data2/dogyun/enhanced_latency_fid_comparison.png", dpi=300, bbox_inches='tight')
plt.show()