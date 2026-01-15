"""Plot experiment results organized by model.

Creates one plot per model, with different actual temperatures as lines.
X-axis: N (number of warmup questions)
Y-axis: reported/guessed temperature
"""

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


def compute_statistics_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and stderr for each model/temperature/n_questions group."""
    # Filter to only successfully parsed results
    df_valid = df[df['parsed_temperature'].notna()].copy()

    # Group by model, actual_temperature, and n_questions
    grouped = df_valid.groupby(['model', 'actual_temperature', 'n_questions'])

    stats = grouped['parsed_temperature'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
    ]).reset_index()

    # Calculate standard error
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])

    return stats


def plot_by_model(stats: pd.DataFrame, output_dir: str = 'results'):
    """Create line plots: one per model, showing reported temp vs N questions for each actual temp."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    actual_temps = sorted(stats['actual_temperature'].unique())
    models = sorted(stats['model'].unique())

    # Colors for actual temperatures
    temp_colors = {0.0: '#2ecc71', 0.5: '#3498db', 1.0: '#e74c3c'}
    temp_labels = {0.0: 'Actual T=0.0', 0.5: 'Actual T=0.5', 1.0: 'Actual T=1.0'}

    for model in models:
        fig, ax = plt.subplots(figsize=(10, 6))

        model_data = stats[stats['model'] == model]

        # Order: T=1.0 first, then T=0.5, then T=0.0
        for actual_temp in [1.0, 0.5, 0.0]:
            temp_data = model_data[model_data['actual_temperature'] == actual_temp]
            temp_data = temp_data.sort_values('n_questions')

            if len(temp_data) == 0:
                continue

            x = temp_data['n_questions'].values
            y = temp_data['mean'].values
            yerr = temp_data['stderr'].values

            ax.errorbar(
                x, y,
                yerr=yerr,
                label=temp_labels[actual_temp],
                color=temp_colors[actual_temp],
                marker='o',
                capsize=4,
                linewidth=2,
                markersize=8,
            )

            # Add horizontal line for actual temperature (reference)
            ax.axhline(y=actual_temp, color=temp_colors[actual_temp], 
                      linestyle='--', alpha=0.3, linewidth=1.5)

        ax.set_xlabel('Number of Warmup Questions (N)', fontsize=12)
        ax.set_ylabel('Average Reported Temperature', fontsize=12)
        ax.set_title(f'Model: {model}\nReported Temperature vs Warmup Questions', fontsize=14)
        ax.set_xticks([0, 1, 2, 4, 8, 16])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save with model name (replace dots with underscores for filename)
        safe_name = model.replace('.', '_').replace('-', '_')
        output_path = f'{output_dir}/model_{safe_name}.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved plot: {output_path}")


def main():
    print("Loading results...")
    results = load_results()
    df = results_to_dataframe(results)

    print(f"Total results: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")

    # Compute grouped statistics
    stats = compute_statistics_by_group(df)

    # Generate plots
    print("\nGenerating plots by model...")
    plot_by_model(stats)

    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
