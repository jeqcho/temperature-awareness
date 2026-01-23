"""Plot histograms for 50/50 word choice experiment with position analysis."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
RESULTS_FILE = "results_logprob_50/experiment_results.json"
OUTPUT_DIR = "plots/logprob_50"


def load_results(filepath: str) -> list[dict]:
    """Load experiment results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_combined_histogram(results: list[dict]):
    """
    Create a single histogram showing distribution of word1 probabilities.
    
    In a 50/50 task, if model is calibrated, prob_word1 should be centered at 0.5.
    """
    # Extract probabilities
    probs_word1 = [r["prob_word1"] for r in results if r.get("prob_word1") is not None]
    
    if not probs_word1:
        print("No valid probability data found!")
        return
    
    print(f"Plotting {len(probs_word1)} data points for combined histogram")
    
    # Create figure - slide quality size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bins from 0 to 1
    bins = np.linspace(0, 1, 25)
    
    # Plot histogram with default color
    ax.hist(probs_word1, bins=bins, alpha=0.7, color="#1f77b4", edgecolor="black", linewidth=0.8)
    
    # Calculate mean
    mean_prob = np.mean(probs_word1)
    std_prob = np.std(probs_word1)
    
    # Add solid vertical line for target value (0.5)
    ax.axvline(x=0.5, color="green", linestyle="-", linewidth=2.5, alpha=0.9, label="Target (0.50)")
    
    # Add dashed vertical line for actual mean
    ax.axvline(x=mean_prob, color="red", linestyle="--", linewidth=2.5, alpha=0.9, label=f"Mean ({mean_prob:.2f})")
    
    # Labels and title
    ax.set_xlabel("Probability of Word1 (from logprobs)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Distribution of Word1 Probability for 50/50 Choice Task (T=1)", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    
    # Set x-axis limits
    ax.set_xlim(0, 1)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Add summary stats as text
    stats_text = f"μ={mean_prob:.3f}, σ={std_prob:.3f}\nn={len(probs_word1)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Save figure
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/probability_histogram_combined.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    
    plt.close()


def plot_position_histogram(results: list[dict]):
    """
    Create overlapping histogram colored by word position.
    
    Shows if there's a position bias (e.g., model always favors first word).
    """
    # Extract probabilities
    probs_word1 = [r["prob_word1"] for r in results if r.get("prob_word1") is not None]
    probs_word2 = [r["prob_word2"] for r in results if r.get("prob_word2") is not None]
    
    if not probs_word1 or not probs_word2:
        print("No valid probability data found!")
        return
    
    print(f"Plotting {len(probs_word1)} data points for position histogram")
    
    # Create figure - slide quality size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bins from 0 to 1
    bins = np.linspace(0, 1, 25)
    
    # Plot overlapping histograms with transparency
    ax.hist(probs_word1, bins=bins, alpha=0.5, label="Word1 (first position)", color="#2563eb", edgecolor="black", linewidth=0.8)
    ax.hist(probs_word2, bins=bins, alpha=0.5, label="Word2 (second position)", color="#f97316", edgecolor="black", linewidth=0.8)
    
    # Calculate means
    mean_word1 = np.mean(probs_word1)
    mean_word2 = np.mean(probs_word2)
    
    # Add solid vertical line for target value (0.5)
    ax.axvline(x=0.5, color="green", linestyle="-", linewidth=2.5, alpha=0.9, label="Target (0.50)")
    
    # Add dashed vertical lines for actual means
    ax.axvline(x=mean_word1, color="#2563eb", linestyle="--", linewidth=2.5, alpha=0.9, label=f"Mean Word1 ({mean_word1:.2f})")
    ax.axvline(x=mean_word2, color="#f97316", linestyle="--", linewidth=2.5, alpha=0.9, label=f"Mean Word2 ({mean_word2:.2f})")
    
    # Labels and title
    ax.set_xlabel("Probability (from logprobs)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Distribution of Probabilities by Word Position for 50/50 Task (T=1)", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    
    # Set x-axis limits
    ax.set_xlim(0, 1)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Add summary stats as text
    std_word1 = np.std(probs_word1)
    std_word2 = np.std(probs_word2)
    
    stats_text = f"Word1: μ={mean_word1:.3f}, σ={std_word1:.3f}\nWord2: μ={mean_word2:.3f}, σ={std_word2:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Save figure
    output_file = f"{OUTPUT_DIR}/probability_histogram_by_position.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    
    plt.close()


def analyze_position_bias(results: list[dict]):
    """
    Analyze position bias and print raw results for high-probability words.
    """
    successful = [r for r in results if r.get("prob_word1") is not None]
    
    if not successful:
        print("No successful results to analyze!")
        return
    
    print("\n" + "=" * 60)
    print("Position Bias Analysis")
    print("=" * 60)
    
    probs_word1 = [r["prob_word1"] for r in successful]
    probs_word2 = [r["prob_word2"] for r in successful]
    
    mean_word1 = np.mean(probs_word1)
    mean_word2 = np.mean(probs_word2)
    std_word1 = np.std(probs_word1)
    std_word2 = np.std(probs_word2)
    
    print(f"\nOverall Statistics:")
    print(f"  Word1 (first position):  μ={mean_word1:.4f}, σ={std_word1:.4f}")
    print(f"  Word2 (second position): μ={mean_word2:.4f}, σ={std_word2:.4f}")
    print(f"  Expected:                μ=0.5000")
    
    # Position bias
    bias = mean_word1 - 0.5
    if abs(bias) < 0.05:
        bias_str = "No significant position bias"
    elif bias > 0:
        bias_str = f"Model favors FIRST position by {bias:.2%}"
    else:
        bias_str = f"Model favors SECOND position by {-bias:.2%}"
    print(f"\n  Position Bias: {bias_str}")
    
    # Find words with extreme probabilities
    print("\n" + "-" * 60)
    print("Words with Highest Probabilities (>0.95)")
    print("-" * 60)
    
    high_prob_word1 = [(r["word1"], r["word2"], r["prob_word1"]) for r in successful if r["prob_word1"] > 0.95]
    high_prob_word2 = [(r["word1"], r["word2"], r["prob_word2"]) for r in successful if r["prob_word2"] > 0.95]
    
    if high_prob_word1:
        print(f"\nWord1 (first position) with prob > 0.95: {len(high_prob_word1)} cases")
        for word1, word2, prob in sorted(high_prob_word1, key=lambda x: -x[2])[:10]:
            print(f"  '{word1}' vs '{word2}': {prob:.4f}")
    else:
        print("\nNo Word1 with prob > 0.95")
    
    if high_prob_word2:
        print(f"\nWord2 (second position) with prob > 0.95: {len(high_prob_word2)} cases")
        for word1, word2, prob in sorted(high_prob_word2, key=lambda x: -x[2])[:10]:
            print(f"  '{word1}' vs '{word2}': {prob:.4f}")
    else:
        print("\nNo Word2 with prob > 0.95")
    
    # Find words that appear multiple times and check if they're consistently high
    print("\n" + "-" * 60)
    print("Raw Results Sample (first 20)")
    print("-" * 60)
    print(f"{'Word1':<15} {'Word2':<15} {'P(Word1)':<10} {'P(Word2)':<10} {'Response':<15}")
    print("-" * 60)
    
    for r in successful[:20]:
        word1 = r["word1"][:14]
        word2 = r["word2"][:14]
        p1 = r["prob_word1"]
        p2 = r["prob_word2"]
        resp = (r.get("response") or "N/A")[:14]
        print(f"{word1:<15} {word2:<15} {p1:<10.4f} {p2:<10.4f} {resp:<15}")
    
    # Distribution summary
    print("\n" + "-" * 60)
    print("Distribution Summary")
    print("-" * 60)
    
    # Bin into ranges
    ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print("\nWord1 probability distribution:")
    for low, high in ranges:
        count = sum(1 for p in probs_word1 if low <= p < high)
        pct = 100 * count / len(probs_word1)
        bar = "█" * int(pct / 2)
        print(f"  [{low:.1f}, {high:.1f}): {count:3d} ({pct:5.1f}%) {bar}")


def main():
    print("=" * 60)
    print("Plotting 50/50 Logprob Histograms")
    print("=" * 60)
    
    # Load results
    results = load_results(RESULTS_FILE)
    print(f"Loaded {len(results)} results from {RESULTS_FILE}")
    
    # Plot combined histogram
    print("\n--- Combined Histogram ---")
    plot_combined_histogram(results)
    
    # Plot position histogram
    print("\n--- Position-Colored Histogram ---")
    plot_position_histogram(results)
    
    # Analyze position bias
    analyze_position_bias(results)


if __name__ == "__main__":
    main()
