"""Plot results from multi-turn strategy temperature experiment.

Creates plots showing:
1. Per-model: reported temperature vs N rounds for each actual temp
2. Accuracy: percentage of correct guesses by model and N rounds
3. Confusion matrix: predicted vs actual temperature distribution
4. Error analysis: L1 and L2 errors by N rounds
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(filename: str = 'src/results_strategy/experiment_results.json') -> list[dict]:
    """Load results from JSON file."""
    with open(filename) as f:
        return json.load(f)


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to pandas DataFrame."""
    return pd.DataFrame(results)


def compute_statistics_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, and stderr for each model/temperature/n_rounds group."""
    # Filter to only successfully parsed results
    df_valid = df[df['parsed_temperature'].notna()].copy()

    # Group by model, actual_temperature, and n_rounds
    grouped = df_valid.groupby(['model', 'actual_temperature', 'n_rounds'])

    stats = grouped['parsed_temperature'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
    ]).reset_index()

    # Calculate standard error
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])

    return stats


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy (exact match) for each model/n_rounds group."""
    df_valid = df[df['parsed_temperature'].notna()].copy()
    
    # Check if prediction is correct (exact match for discrete choices)
    df_valid['correct'] = df_valid['parsed_temperature'] == df_valid['actual_temperature']
    
    # Group by model and n_rounds
    grouped = df_valid.groupby(['model', 'n_rounds'])
    
    accuracy = grouped.agg(
        accuracy=('correct', 'mean'),
        correct_count=('correct', 'sum'),
        total_count=('correct', 'count'),
    ).reset_index()
    
    # Calculate standard error for proportion
    accuracy['stderr'] = np.sqrt(
        accuracy['accuracy'] * (1 - accuracy['accuracy']) / accuracy['total_count']
    )
    
    return accuracy


def compute_error_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute L1 and L2 error statistics grouped by model and n_rounds."""
    df_valid = df[df['parsed_temperature'].notna()].copy()
    
    # Compute errors
    df_valid['l1_error'] = (df_valid['parsed_temperature'] - df_valid['actual_temperature']).abs()
    df_valid['l2_error'] = (df_valid['parsed_temperature'] - df_valid['actual_temperature']) ** 2
    
    # Group by model and n_rounds
    grouped = df_valid.groupby(['model', 'n_rounds'])
    
    stats = grouped.agg(
        l1_mean=('l1_error', 'mean'),
        l1_std=('l1_error', 'std'),
        l2_mean=('l2_error', 'mean'),
        l2_std=('l2_error', 'std'),
        count=('l1_error', 'count'),
    ).reset_index()
    
    stats['l1_stderr'] = stats['l1_std'] / np.sqrt(stats['count'])
    stats['l2_stderr'] = stats['l2_std'] / np.sqrt(stats['count'])
    
    return stats


def compute_confusion_matrix(df: pd.DataFrame) -> dict:
    """Compute confusion matrix data per model."""
    df_valid = df[df['parsed_temperature'].notna()].copy()
    
    models = df_valid['model'].unique()
    actual_temps = [0.0, 0.5, 1.0]
    
    confusion_data = {}
    for model in models:
        model_df = df_valid[df_valid['model'] == model]
        matrix = np.zeros((3, 3))
        
        for i, actual in enumerate(actual_temps):
            for j, predicted in enumerate(actual_temps):
                count = len(model_df[
                    (model_df['actual_temperature'] == actual) & 
                    (model_df['parsed_temperature'] == predicted)
                ])
                matrix[i, j] = count
        
        # Normalize by row (actual temperature)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix_normalized = matrix / row_sums
        
        confusion_data[model] = {
            'raw': matrix,
            'normalized': matrix_normalized,
        }
    
    return confusion_data


def plot_by_model(stats: pd.DataFrame, output_dir: str = 'plots/strategy'):
    """Create line plots: one per model, showing reported temp vs N rounds for each actual temp."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = sorted(stats['model'].unique())

    # Colors for actual temperatures
    temp_colors = {0.0: '#2ecc71', 0.5: '#3498db', 1.0: '#e74c3c'}
    temp_labels = {0.0: 'Actual T=0.0', 0.5: 'Actual T=0.5', 1.0: 'Actual T=1.0'}

    for model in models:
        fig, ax = plt.subplots(figsize=(12, 7))

        model_data = stats[stats['model'] == model]

        # Order: T=1.0 first, then T=0.5, then T=0.0
        for actual_temp in [1.0, 0.5, 0.0]:
            temp_data = model_data[model_data['actual_temperature'] == actual_temp]
            temp_data = temp_data.sort_values('n_rounds')

            if len(temp_data) == 0:
                continue

            x = temp_data['n_rounds'].values
            y = temp_data['mean'].values
            yerr = temp_data['stderr'].values

            ax.errorbar(
                x, y,
                yerr=yerr,
                label=temp_labels[actual_temp],
                color=temp_colors[actual_temp],
                marker='o',
                capsize=5,
                linewidth=2.5,
                markersize=10,
            )

            # Add horizontal line for actual temperature (reference)
            ax.axhline(y=actual_temp, color=temp_colors[actual_temp], 
                      linestyle='--', alpha=0.3, linewidth=2)

        ax.set_xlabel('Number of Strategy Rounds (N)', fontsize=16)
        ax.set_ylabel('Average Reported Temperature', fontsize=16)
        ax.set_title(f'Model: {model}\nReported Temperature vs Strategy Rounds', fontsize=18)
        ax.set_xticks([0, 1, 2, 4, 8, 16])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save with model name (replace dots with underscores for filename)
        safe_name = model.replace('.', '_').replace('-', '_')
        output_path = f'{output_dir}/model_{safe_name}.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved plot: {output_path}")


def plot_accuracy(accuracy: pd.DataFrame, output_dir: str = 'plots/strategy'):
    """Plot accuracy by model and N rounds."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = sorted(accuracy['model'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in models:
        model_data = accuracy[accuracy['model'] == model].sort_values('n_rounds')
        
        if len(model_data) == 0:
            continue
        
        x = model_data['n_rounds'].values
        y = model_data['accuracy'].values * 100  # Convert to percentage
        yerr = model_data['stderr'].values * 100
        
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
    
    # Random baseline (33.3%)
    ax.axhline(y=33.3, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Random (33.3%)')
    
    ax.set_xlabel('Number of Strategy Rounds (N)', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_title('Temperature Prediction Accuracy (Strategy Experiment)\nvs Strategy Rounds', fontsize=18)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/accuracy_vs_n_rounds.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_errors(error_stats: pd.DataFrame, output_dir: str = 'plots/strategy'):
    """Plot L1 and L2 errors by model and N rounds."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = sorted(error_stats['model'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    for error_type in ['l1', 'l2']:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        mean_col = f'{error_type}_mean'
        stderr_col = f'{error_type}_stderr'
        
        for model in models:
            model_data = error_stats[error_stats['model'] == model].sort_values('n_rounds')
            
            if len(model_data) == 0:
                continue
            
            x = model_data['n_rounds'].values
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
        
        error_label = 'L1 (Mean Absolute Error)' if error_type == 'l1' else 'L2 (Mean Squared Error)'
        ax.set_xlabel('Number of Strategy Rounds (N)', fontsize=16)
        ax.set_ylabel(f'Mean {error_label}', fontsize=16)
        ax.set_title(f'{error_label} vs Strategy Rounds', fontsize=18)
        ax.set_xticks([0, 1, 2, 4, 8, 16])
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{output_dir}/{error_type}_error_vs_n_rounds.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved plot: {output_path}")


def plot_confusion_matrices(confusion_data: dict, output_dir: str = 'plots/strategy'):
    """Plot confusion matrices for each model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    temp_labels = ['0.0', '0.5', '1.0']
    
    for model, data in confusion_data.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        
        matrix = data['normalized']
        
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Proportion', rotation=-90, va='bottom', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                raw_count = int(data['raw'][i, j])
                pct = matrix[i, j] * 100
                text = f'{pct:.1f}%\n(n={raw_count})'
                color = 'white' if matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=12)
        
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(temp_labels, fontsize=14)
        ax.set_yticklabels(temp_labels, fontsize=14)
        ax.set_xlabel('Predicted Temperature', fontsize=16)
        ax.set_ylabel('Actual Temperature', fontsize=16)
        ax.set_title(f'Model: {model}\nConfusion Matrix (Strategy Experiment)', fontsize=18)
        
        plt.tight_layout()
        
        safe_name = model.replace('.', '_').replace('-', '_')
        output_path = f'{output_dir}/confusion_{safe_name}.png'
        plt.savefig(output_path, dpi=300)
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

    # Generate plots by model
    print("\nGenerating plots by model...")
    plot_by_model(stats)
    
    # Compute and plot accuracy
    print("\nComputing accuracy...")
    accuracy = compute_accuracy(df)
    print("\nAccuracy by model and N rounds:")
    print(accuracy.to_string(index=False))
    plot_accuracy(accuracy)
    
    # Compute and plot errors
    print("\nComputing error statistics...")
    error_stats = compute_error_statistics(df)
    plot_errors(error_stats)
    
    # Compute and plot confusion matrices
    print("\nGenerating confusion matrices...")
    confusion_data = compute_confusion_matrix(df)
    plot_confusion_matrices(confusion_data)

    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
