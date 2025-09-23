import matplotlib.pyplot as plt
import numpy as np

# Training speed data (steps/sec)
methods = ['LeRobot Original', 'RoboDM Dataset']
speeds = [11.81, 19.52]  # Example training speeds in steps/sec

# Create the bar chart
plt.figure(figsize=(8, 8))
bars = plt.bar(methods, speeds, color=['#1f77b4', '#ff7f0e'], width=0.4)

# Customize the chart
plt.title('Training Speed Comparison', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Training Method', fontsize=16)
plt.ylabel('Training Speed (steps/sec)', fontsize=16)

# Add value labels on top of each bar
for bar, speed in zip(bars, speeds):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{speed} steps/sec',
             ha='center', va='bottom', fontweight='bold')

# Customize the y-axis
plt.ylim(0, max(speeds) * 1.2)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Make axis tick labels larger
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Keep x-axis labels horizontal
plt.xticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Optional: Save the plot
plt.savefig('training_speed_comparison.png', dpi=300, bbox_inches='tight')
