import matplotlib.pyplot as plt
import numpy as np

# Example Data: [GFLOPs, FID]
methods = {
    "Flow Matching (Baseline)": [(54.56*3, 55.34), (54.56*7, 21.69), (54.56*12, 15.78), (54.56*24, 12.34), (54.56*36, 11.72)],
    #"BFM-XL-Align (Ours)": [(174.5, 12.61), (174.5*2, 10.3), (174.5*3, 9.6), (174.5*4, 9.26), (174.5*5, 8.97)],
    "BFM-B-Align (Ours)": [(98.23, 13.7), (98.23*2, 10.59), (98.23*3, 9.6), (98.23*4, 9.16), (98.23*5, 8.88), (98.23*6, 8.72)],
}

# Define Colors and Markers
colors = ["red", "blue"]
markers = ["o", "^"]

# Plot Each Method
plt.figure(figsize=(8,6))
for i, (method, points) in enumerate(methods.items()):
    # Convert points to numpy arrays for easier handling
    points = np.array(points)
    latencies = points[:, 0]
    fids = points[:, 1]
    
    # Plot points
    #plt.plot(latencies, fids, color=colors[i], linestyle='-', alpha=0.5)
    plt.semilogy(latencies, fids, color=colors[i], linestyle='-', alpha=0.5)  # Changed to semilogy for log scale
    plt.scatter(latencies, fids, color=colors[i], marker=markers[i], s=150, label=method)
    
    # Add labels for each point
    for lat, fid in zip(latencies, fids):
        plt.text(lat+5, fid, f"{fid:.2f}", fontsize=8, ha="left")

# Labels and Formatting
plt.xlabel("GFLOPs ", fontsize=12)
plt.ylabel("FID ↓ (Lower is Better)", fontsize=12)
plt.title("Inference Complexity vs. FID Trade-off", fontsize=14)
plt.legend()
plt.grid(True, linestyle="dotted")
plt.savefig("/hub_data2/dogyun/LWD/fitv2_lwd_cifar_latency_fid.png")