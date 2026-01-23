"""Plot L1 and L2 errors across all temperatures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(filename: str = 'results/experiment_results.json') -> list[dict]:
    """Load results from JSON file."""
    with open(filename) as f:
        return json.load(f)


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to pandas DataFrame."""
    return pd.DataFrame(results)


def compute_error_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute L1 and L2 error statistics grouped by model and n_questions.
    
    Aggregates across all temperature settings.
    """
    # Filter to only successfully parsed results
    df_valid = df[df['parsed_temperature'].notna()].copy()
    
    # Compute errors
    df_valid['l1_error'] = (df_valid['parsed_temperature'] - df_valid['actual_temperature']).abs()
    df_valid['l2_error'] = (df_valid['parsed_temperature'] - df_valid['actual_temperature']) ** 2
    
    # Group by model and n_questions (aggregating across all temperatures)
    grouped = df_valid.groupby(['model', 'n_questions'])
    
    stats = grouped.agg(
        l1_mean=('l1_error', 'mean'),
        l1_std=('l1_error', 'std'),
        l2_mean=('l2_error', 'mean'),
        l2_std=('l2_error', 'std'),
        count=('l1_error', 'count'),
    ).reset_index()
    
    # Calculate standard errors
    stats['l1_stderr'] = stats['l1_std'] / np.sqrt(stats['count'])
    stats['l2_stderr'] = stats['l2_std'] / np.sqrt(stats['count'])
    
    return stats


def plot_error(
    stats: pd.DataFrame,
    error_type: str,
    output_dir: str = 'plots/main',
) -> None:
    """Create a line plot for the specified error type.
    
    Args:
        stats: DataFrame with error statistics
        error_type: 'l1' or 'l2'
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = sorted(stats['model'].unique())
    
    # Color map for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    mean_col = f'{error_type}_mean'
    stderr_col = f'{error_type}_stderr'
    
    for model in models:
        model_data = stats[stats['model'] == model].sort_values('n_questions')
        
        if len(model_data) == 0:
            continue
        
        x = model_data['n_questions'].values
        y = model_data[mean_col].values
        yerr = model_data[stderr_col].values
        
        ax.errorbar(
            x, y,
            yerr=yerr,
            label=model,
            color=model_colors[model],
            marker='o',
            capsize=5,
            linewidth=2.5,
            markersize=10,
        )
    
    # Labels and formatting
    error_label = 'L1 (Mean Absolute Error)' if error_type == 'l1' else 'L2 (Mean Squared Error)'
    ax.set_xlabel('Number of Warmup Questions (N)', fontsize=16)
    ax.set_ylabel(f'Mean {error_label}', fontsize=16)
    ax.set_title(f'{error_label} vs Warmup Questions\n(Combined Across All Temperatures)', fontsize=18)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/{error_type}_error_vs_n_questions.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    print("Loading results...")
    results = load_results()
    df = results_to_dataframe(results)
    
    print(f"Total results: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")
    
    # Compute error statistics
    print("\nComputing error statistics...")
    stats = compute_error_statistics(df)
    
    # Print summary
    print("\nError Statistics Summary:")
    print(stats.to_string(index=False))
    
    # Generate plots
    print("\nGenerating error plots...")
    plot_error(stats, 'l1')
    plot_error(stats, 'l2')
    
    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
