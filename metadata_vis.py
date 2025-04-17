import os
import numpy as np
import matplotlib.pyplot as plt
import json

with open("model_layer_metadata.json", "r") as meta_file:
    data = json.loads(meta_file.read())

if not os.path.exists('plot_dir'):
    os.mkdir('plot_dir')

# Convert keys to ints and sort
depths = sorted(int(k) for k in data.keys())
labels = ['conv', 'dense', 'activation', 'pooling']

# Get values for each layer type per depth
layer_counts = {label: [data[str(depth)][label] / 1000 for depth in depths] for label in labels}

# Bar width and position
x = np.arange(len(depths))
bar_width = 0.2

""" Combined layer counts """
fig, ax = plt.subplots(figsize=(10, 6))

for i, label in enumerate(labels):
    ax.bar(x + i * bar_width, layer_counts[label], width=bar_width, label=label)

ax.set_xticks(x + bar_width * (len(labels) - 1) / 2)
ax.set_xticklabels(depths)
ax.set_xlabel('Model Depth')
ax.set_ylabel('Layer Count')
ax.set_title('Layer Counts per Model Depth')
ax.legend()
plt.tight_layout()
plt.savefig("plot_dir/combined_histogram.png", dpi=300, bbox_inches="tight")

""" Per layer count loop"""
layer_types = ['conv', 'dense', 'activation', 'pooling']
sorted_depths = sorted(map(int, data.keys()))

# Plot one histogram per layer type
for layer in layer_types:
    counts = [data[str(d)][layer] for d in sorted_depths]

    plt.figure(figsize=(8, 5))
    plt.bar(sorted_depths, counts, color='red')
    plt.xlabel('Model Layer Count')
    plt.ylabel(f'Total {layer} layers')
    plt.title(f'{layer.capitalize()} Layers by Model Depth')
    plt.xticks(sorted_depths)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plot_dir/{layer}_histogram.png", dpi=300, bbox_inches="tight")