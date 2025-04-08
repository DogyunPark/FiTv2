import matplotlib.pyplot as plt
import numpy as np

# Example Data: [GFLOPs, FID]
methods = {
    "Flow Matching (Baseline)": [(0.1511*1000, 55.34), (0.2336*1000, 33.19), (0.3229*1000, 21.69), (0.5951*1000, 15.78), (1.0672*1000, 12.34), (1.6457*1000, 11.72)],
    "BFM-L-Align (Ours)": [(174.5, 12.61), (174.5*2, 10.3), (174.5*3, 9.6), (174.5*4, 9.26), (174.5*5, 8.97)],
    #"BFM-L (Ours)": [(0.1511*1000, 55.34), (0.2336*1000, 33.19), (0.3229*1000, 21.69), (0.5951*1000, 15.78), (1.0672*1000, 12.34), (1.6457*1000, 11.72)],
    "BFM-B-Align (Ours)": [(0.4444*1000, 13.7), (0.5608*1000, 10.59), (0.8287*1000, 9.6), (1.0830*1000, 9.16), (1.2682*1000, 8.88), (1.5020*1000, 8.72)],
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
plt.xlabel("Latency (ms per image)", fontsize=12)
plt.ylabel("FID ↓ (Lower is Better)", fontsize=12)
plt.title("Inference Complexity vs. FID Trade-off", fontsize=14)
plt.legend()
plt.grid(True, linestyle="dotted")
plt.savefig("/hub_data2/dogyun/LWD/fitv2_lwd_cifar_latency_fid_2.png")