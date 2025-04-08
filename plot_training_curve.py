import matplotlib.pyplot as plt
import numpy as np

# Example Data: [GFLOPs, FID]
Iterations = [
    2500,
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    75000,
    80000,
    85000,
    90000,
    95000,
    100000,
]

methods_FID = {
    "Flow Matching (Baseline)": [
        282-40,
        188-40,
        143-40,
        131-40,
        121-40,
        116-40,
        113-40,
        112-40,
        111-40,
        111-40,
        108.9-40,
        108-40,
        108.9-40,
        108.8-40,
        107-40,
        105-40,
        104-40,
        103-40,
        101-40,
        101.85-40,
        104-40,
    ],
    "BFM-S (Ours)": [
        209,
        102,
        59,
        46,
        43,
        41,
        39,
        38,
        37,
        36,
        35.5,
        35,
        34.5,
        34,
        33.5,
        32.9,
        32,
        31.6,
        31,
        30.5,
        30,
    ],
    "BFM-S + SFC (Ours)": [
        154,
        79.4,
        45.1,
        39.3,
        35.4,
        32,
        30,
        27.8,
        25.8,
        25,
        24.2,
        23,
        22,
        21.7,
        21.43,
        20.7,
        19,
        18.5,
        18.3,
        18,
        18,
    ],
    "BFM-L + SFC (Ours)": [
        101,
        54.82,
        24.32,
        19.21,
        16.31,
        14.8,
        14.2,
        13.2,
        12.78,
        12.23,
        11.9,
        11.6,
        11.62,
        11.72,
        11.7,
        11.54,
        11.6,
        11.4,
        11.45,
        11.3,
        11.2,
    ],
    "BFM-L + SFC + TA (Ours)": [
        69.9,
        41.8,
        21.3,
        16.6,
        14.5,
        12.8,
        11.87,
        11,
        11,
        10.47,
        10.58,
    ],
}


def select_regular_intervals(iterations, values):
    # Define intervals that make sense for the data range (adjust these as needed)
    key_points = [2500, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    # Find closest available points to our desired intervals
    selected_indices = []
    for point in key_points:
        if point <= iterations[-1]:  # Only include points within our data range
            # Find the index of the closest value
            idx = min(range(len(iterations)), key=lambda i: abs(iterations[i] - point))
            if idx not in selected_indices:
                selected_indices.append(idx)
    
    # Return selected iterations and values
    return [iterations[i] for i in selected_indices], [values[i] for i in selected_indices]

# Set figure size and style
plt.figure(figsize=(8, 6))
# try:
#     # Try to import seaborn for better styling
#     import seaborn as sns
#     sns.set_style("whitegrid")
# except ImportError:
#     # Fall back to a built-in matplotlib style if seaborn is not available
#     plt.style.use('ggplot')  # 'ggplot' is a built-in style similar to seaborn

# Colors for different methods
#colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47', '#5B9BD5']
colors = ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099']
#markers = ['o', '^', 's', 'D', 'P']
markers = ['o', 's', 'D', 'P', '*']
#markers = ['o', 's', '^', 'd', 'v']
#markers = ['o', 'h', 'p', 'X', 'd']

# Plot each method
for i, (method, fid_scores) in enumerate(methods_FID.items()):
    # Only use the number of iterations that match the length of FID scores
    iterations = Iterations[:len(fid_scores)]
    smooth_iters, smooth_scores = select_regular_intervals(iterations, fid_scores)
    plt.plot(smooth_iters, smooth_scores, linestyle='-', alpha=0.3, color=colors[i])
    plt.scatter(smooth_iters, smooth_scores, color=colors[i], marker=markers[i], s=50, label=method, alpha=0.6)

# Customize the plot
#plt.xscale('log')  # Use log scale for iterations
plt.yscale('log')  # Use log scale for FID scores
#plt.yticks([0, 50, 100, 150, 200, 250, 300])
plt.xticks([10000, 20000, 30000, 40000, 50000], 
           ['10K', '20K', '30K', '40K', '50K'])
plt.grid(True, which="both", ls="-", alpha=0.5)

# Labels and title
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('FID Score', fontsize=12)
plt.title('Training Curves: FID Score vs Iterations', fontsize=14, pad=20)

# Legend
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
# Save the plot to file
plt.savefig("/hub_data2/dogyun/training_curves_fid_vs_iterations.png", dpi=300, bbox_inches='tight')



