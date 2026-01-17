"""Plot overlapping histograms for 60% vs 40% target probability distributions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_FILE = "results_logprob/experiment_results.json"
OUTPUT_DIR = "results_logprob"


def load_results(filepath: str) -> list[dict]:
    """Load experiment results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_histogram(results: list[dict]):
    """
    Create overlapping histogram of 60% vs 40% target probabilities.
    
    Single plot for T=1 with two semi-transparent distributions.
    """
    # Extract probabilities
    probs_60 = [r["prob_60_target"] for r in results if r.get("prob_60_target") is not None]
    probs_40 = [r["prob_40_target"] for r in results if r.get("prob_40_target") is not None]
    
    if not probs_60 or not probs_40:
        print("No valid probability data found!")
        return
    
    print(f"Plotting {len(probs_60)} data points for 60% target")
    print(f"Plotting {len(probs_40)} data points for 40% target")
    
    # Create figure - slide quality size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define bins from 0 to 1
    bins = np.linspace(0, 1, 25)
    
    # Plot overlapping histograms with transparency
    ax.hist(probs_60, bins=bins, alpha=0.5, label="60% target", color="#2563eb", edgecolor="black", linewidth=0.5)
    ax.hist(probs_40, bins=bins, alpha=0.5, label="40% target", color="#f97316", edgecolor="black", linewidth=0.5)
    
    # Calculate means
    mean_60 = np.mean(probs_60)
    mean_40 = np.mean(probs_40)
    
    # Add solid vertical lines for target values (0.6 and 0.4)
    ax.axvline(x=0.6, color="#2563eb", linestyle="-", linewidth=2, alpha=0.9, label="Target 60%")
    ax.axvline(x=0.4, color="#f97316", linestyle="-", linewidth=2, alpha=0.9, label="Target 40%")
    
    # Add dashed vertical lines for actual means achieved
    ax.axvline(x=mean_60, color="#2563eb", linestyle="--", linewidth=2, alpha=0.9, label=f"Mean 60% ({mean_60:.2f})")
    ax.axvline(x=mean_40, color="#f97316", linestyle="--", linewidth=2, alpha=0.9, label=f"Mean 40% ({mean_40:.2f})")
    
    # Labels and title
    ax.set_xlabel("Probability (from logprobs)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Model Probabilities for 60/40 Word Choice Task (T=1)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    
    # Set x-axis limits
    ax.set_xlim(0, 1)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Add summary stats as text
    std_60 = np.std(probs_60)
    std_40 = np.std(probs_40)
    
    stats_text = f"60% target: μ={mean_60:.3f}, σ={std_60:.3f}\n40% target: μ={mean_40:.3f}, σ={std_40:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Save figure
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/probability_histogram_T1.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_file}")
    
    plt.close()


def main():
    print("=" * 60)
    print("Plotting Logprob Histogram")
    print("=" * 60)
    
    # Load results
    results = load_results(RESULTS_FILE)
    print(f"Loaded {len(results)} results from {RESULTS_FILE}")
    
    # Plot histogram
    plot_histogram(results)
    
    # Print detailed stats
    probs_60 = [r["prob_60_target"] for r in results if r.get("prob_60_target") is not None]
    probs_40 = [r["prob_40_target"] for r in results if r.get("prob_40_target") is not None]
    
    if probs_60:
        print("\nDetailed Statistics:")
        print(f"  60% target - Mean: {np.mean(probs_60):.4f}, Std: {np.std(probs_60):.4f}")
        print(f"  60% target - Min: {np.min(probs_60):.4f}, Max: {np.max(probs_60):.4f}")
        print(f"  40% target - Mean: {np.mean(probs_40):.4f}, Std: {np.std(probs_40):.4f}")
        print(f"  40% target - Min: {np.min(probs_40):.4f}, Max: {np.max(probs_40):.4f}")


if __name__ == "__main__":
    main()
