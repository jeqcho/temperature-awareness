"""Analyze experiment results and generate visualizations."""

import json
from pathlib import Path
from collections import defaultdict

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


def plot_temperature_vs_n_questions(stats: pd.DataFrame, output_dir: str = 'results'):
    """Create line plots: one per actual temperature, showing reported temp vs N questions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    actual_temps = sorted(stats['actual_temperature'].unique())
    models = sorted(stats['model'].unique())

    # Color map for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))

    for actual_temp in actual_temps:
        fig, ax = plt.subplots(figsize=(10, 6))

        temp_data = stats[stats['actual_temperature'] == actual_temp]

        for model in models:
            model_data = temp_data[temp_data['model'] == model].sort_values('n_questions')

            if len(model_data) == 0:
                continue

            x = model_data['n_questions'].values
            y = model_data['mean'].values
            yerr = model_data['stderr'].values

            ax.errorbar(
                x, y,
                yerr=yerr,
                label=model,
                color=model_colors[model],
                marker='o',
                capsize=4,
                linewidth=2,
                markersize=8,
            )

        # Add horizontal line for actual temperature
        ax.axhline(y=actual_temp, color='gray', linestyle='--', alpha=0.5, label=f'Actual ({actual_temp})')

        ax.set_xlabel('Number of Warmup Questions (N)', fontsize=12)
        ax.set_ylabel('Average Reported Temperature', fontsize=12)
        ax.set_title(f'Reported Temperature vs Warmup Questions\n(Actual Temperature = {actual_temp})', fontsize=14)
        ax.set_xticks([0, 1, 2, 4, 8, 16])
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/temp_{actual_temp}_vs_n_questions.png', dpi=150)
        plt.close()
        print(f"Saved plot: {output_dir}/temp_{actual_temp}_vs_n_questions.png")


def print_summary_table(stats: pd.DataFrame):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Average Reported Temperature by Model, Actual Temp, and N")
    print("=" * 80)

    # Pivot table for display
    pivot = stats.pivot_table(
        values='mean',
        index=['model', 'actual_temperature'],
        columns='n_questions',
        aggfunc='first'
    ).round(3)

    print(pivot.to_string())
    print()


def print_parse_success_rate(df: pd.DataFrame):
    """Print parse success rate by model."""
    print("\n" + "=" * 80)
    print("PARSE SUCCESS RATE BY MODEL")
    print("=" * 80)

    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]
        total = len(model_data)
        success = model_data['parsed_temperature'].notna().sum()
        rate = 100 * success / total if total > 0 else 0
        print(f"{model}: {success}/{total} ({rate:.1f}%)")


def compute_correlation(df: pd.DataFrame):
    """Compute correlation between actual and reported temperature."""
    print("\n" + "=" * 80)
    print("CORRELATION: Actual vs Reported Temperature by Model")
    print("=" * 80)

    df_valid = df[df['parsed_temperature'].notna()].copy()

    for model in sorted(df_valid['model'].unique()):
        model_data = df_valid[df_valid['model'] == model]
        if len(model_data) > 1:
            corr = model_data['actual_temperature'].corr(model_data['parsed_temperature'])
            print(f"{model}: r = {corr:.3f}")
        else:
            print(f"{model}: insufficient data")


def main():
    print("Loading results...")
    results = load_results()
    df = results_to_dataframe(results)

    print(f"Total results: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")

    # Print basic stats
    print_parse_success_rate(df)
    compute_correlation(df)

    # Compute grouped statistics
    stats = compute_statistics_by_group(df)
    print_summary_table(stats)

    # Generate plots
    print("\nGenerating plots...")
    plot_temperature_vs_n_questions(stats)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
