import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
df= pd.read_csv("results.csv")
x = df['Step']
y_1 = df['with residual']
y_2 = df['wo residual']
plt.figure(figsize=(6, 6))
plt.plot(x, y_2, linewidth=2, color='gray', alpha=0.3)
plt.plot(x, y_1, linewidth=2, color='blue', alpha=0.3)
# Plot trend lines using rolling window
window_size = 100
y1_rolling = pd.Series(y_1).rolling(window=window_size).mean()
y2_rolling = pd.Series(y_2).rolling(window=window_size).mean()
plt.plot(x, y2_rolling, label='w.o Residual', linewidth=2, color='black')
plt.plot(x, y1_rolling, label='w Residual', linewidth=2, color='blue')
plt.xlabel('Training Steps', ha='center', fontsize=20, fontweight='bold', color="black")
plt.ylabel('$\mathcal{L}_{\t{SRN}}$', ha='center', fontsize=23, fontweight='bold')
# plt.title('w/ Residual vs w/o Residual', fontsize=14, fontweight='bold')
plt.legend(fontsize=18, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray', loc='best')
# Set y-axis limits to better show convergence
plt.ylim(-0.99, -0.95)
index_offset = 100
index_offset_2 = 16000
plt.xlim(index_offset, index_offset_2)
# Only show critical ticks
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
# Add grid lines for ticks
plt.grid(True, linestyle='--', alpha=0.7)
# Save the plot
plt.savefig('residual_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('residual_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()