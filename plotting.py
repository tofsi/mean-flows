import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_scalability(
    checkpoint_dirs: Dict[str, str],
    output_path: str = "scalability_plot.png",
    fid_key: str = "fid_1k",
    epoch_key: str = "epoch",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    title: str = "Scalability of MeanFlow models on ImageNet 256×256",
):
    """
    Plot 1-NFE FID vs training epochs for multiple model architectures.
    
    Args:
        checkpoint_dirs: Dict mapping architecture names to checkpoint directories
                        e.g., {"B/2": "checkpoints/B2", "M/2": "checkpoints/M2"}
        output_path: Where to save the plot
        fid_key: Key in metrics.jsonl for FID values
        epoch_key: Key in metrics.jsonl for epoch numbers
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
        title: Plot title
    """
    
    # Architecture ordering and styling
    arch_order = ["B/2", "M/2", "L/2"]
    
    # Color scheme matching the paper's figure
    colors = {
        "B/2": "#5B8DBE",   # Blue
        "M/2": "#E67E50",   # Orange
        "L/2": "#5FA67C",   # Green
    }
    
    print(f"Plotting scalability to {output_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Track data for annotations
    final_points = {}
    
    # Plot each architecture
    for arch_name in arch_order:
        if arch_name not in checkpoint_dirs:
            print(f"Warning: {arch_name} not found in checkpoint_dirs, skipping")
            continue
            
        checkpoint_dir = Path(checkpoint_dirs[arch_name])
        metrics_file = checkpoint_dir / "metrics.jsonl"
        
        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found, skipping {arch_name}")
            continue
        
        # Load metrics
        epochs = []
        fids = []

        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics = json.loads(line)
                    if fid_key in metrics and epoch_key in metrics:
                        epoch = int(metrics[epoch_key]) + 1
                        fid = metrics[fid_key]
                        epochs.append(epoch)
                        fids.append(fid)
        
        if not epochs:
            print(f"Warning: No data found for {arch_name}")
            continue
        
        # Sort by epoch (in case they're out of order)
        sorted_data = sorted(zip(epochs, fids))
        epochs, fids = zip(*sorted_data)
        
        # Plot line with markers
        line = ax.plot(
            epochs, 
            fids, 
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=colors.get(arch_name, None),
            label=arch_name,
            alpha=0.9
        )[0]
        
        # Store final point for annotation
        if epochs:
            final_points[arch_name] = (epochs[-1], fids[-1])
    
    # Annotate final FID values
    for arch_name, (epoch, fid) in final_points.items():
        ax.annotate(
            f'{fid:.2f}',
            xy=(epoch, fid),
            xytext=(8, 0),
            textcoords='offset points',
            fontsize=10,
            color=colors.get(arch_name, 'black'),
            verticalalignment='center',
            weight='bold'
        )
    
    # Styling
    ax.set_xlabel('Training Epochs', fontsize=13, weight='bold')
    ax.set_ylabel('1-NFE FID', fontsize=13, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend with parameter counts
    legend_labels = [f"{name}" for name in arch_order if name in final_points]
    ax.legend(
        legend_labels,
        loc='upper left',
        framealpha=0.95,
        fontsize=11,
        title='Architecture',
        title_fontsize=11
    )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Scalability Plot saved to {output_path}")
    plt.close()
    
    return fig


    