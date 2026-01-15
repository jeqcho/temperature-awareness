"""Plot results from binary choice temperature experiment (0 or 1 only).

Creates plots showing:
1. Per-model: reported temperature vs N questions with lines for T=0 and T=1
2. Accuracy: percentage of correct guesses by model and N
3. Confusion matrix: 2x2 predicted vs actual temperature distribution
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(filename: str = 'results_choice_binary/experiment_results.json') -> list[dict]:
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


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy (exact match) for each model/n_questions group."""
    df_valid = df[df['parsed_temperature'].notna()].copy()
    
    # Check if prediction is correct (exact match for discrete choices)
    df_valid['correct'] = df_valid['parsed_temperature'] == df_valid['actual_temperature']
    
    # Group by model and n_questions
    grouped = df_valid.groupby(['model', 'n_questions'])
    
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


def compute_confusion_matrix(df: pd.DataFrame) -> dict:
    """Compute 2x2 confusion matrix data per model."""
    df_valid = df[df['parsed_temperature'].notna()].copy()
    
    models = df_valid['model'].unique()
    actual_temps = [0.0, 1.0]  # Binary only
    
    confusion_data = {}
    for model in models:
        model_df = df_valid[df_valid['model'] == model]
        matrix = np.zeros((2, 2))
        
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


def plot_by_model(stats: pd.DataFrame, output_dir: str = 'results_choice_binary'):
    """Create line plots: one per model, showing reported temp vs N questions for T=0 and T=1."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = sorted(stats['model'].unique())

    # Colors for actual temperatures (binary)
    temp_colors = {0.0: '#2ecc71', 1.0: '#e74c3c'}  # Green for 0, Red for 1
    temp_labels = {0.0: 'Actual T=0', 1.0: 'Actual T=1'}

    for model in models:
        fig, ax = plt.subplots(figsize=(10, 6))

        model_data = stats[stats['model'] == model]

        # Order: T=1 first, then T=0
        for actual_temp in [1.0, 0.0]:
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

        # Add horizontal reference lines at y=0 and y=1
        ax.axhline(y=0.0, color=temp_colors[0.0], linestyle='--', alpha=0.3, linewidth=1.5)
        ax.axhline(y=1.0, color=temp_colors[1.0], linestyle='--', alpha=0.3, linewidth=1.5)

        ax.set_xlabel('Number of Warmup Questions (N)', fontsize=12)
        ax.set_ylabel('Average Reported Temperature', fontsize=12)
        ax.set_title(f'Model: {model}\nReported Temperature vs Warmup Questions (Binary: 0 or 1)', fontsize=14)
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


def plot_accuracy(accuracy: pd.DataFrame, output_dir: str = 'results_choice_binary'):
    """Plot accuracy by model and N questions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = sorted(accuracy['model'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        model_data = accuracy[accuracy['model'] == model].sort_values('n_questions')
        
        if len(model_data) == 0:
            continue
        
        x = model_data['n_questions'].values
        y = model_data['accuracy'].values * 100  # Convert to percentage
        yerr = model_data['stderr'].values * 100
        
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
    
    # Random baseline (50% for binary)
    ax.axhline(y=50.0, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    
    ax.set_xlabel('Number of Warmup Questions (N)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Temperature Prediction Accuracy (Binary: 0 or 1)\nvs Warmup Questions', fontsize=14)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_ylim(0, 100)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/accuracy_vs_n_questions.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_confusion_matrices(confusion_data: dict, output_dir: str = 'results_choice_binary'):
    """Plot 2x2 confusion matrices for each model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    temp_labels = ['0', '1']  # Binary only
    
    for model, data in confusion_data.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        
        matrix = data['normalized']
        
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Proportion', rotation=-90, va='bottom')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                raw_count = int(data['raw'][i, j])
                pct = matrix[i, j] * 100
                text = f'{pct:.1f}%\n(n={raw_count})'
                color = 'white' if matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=12)
        
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(temp_labels)
        ax.set_yticklabels(temp_labels)
        ax.set_xlabel('Predicted Temperature', fontsize=12)
        ax.set_ylabel('Actual Temperature', fontsize=12)
        ax.set_title(f'Model: {model}\nConfusion Matrix (Binary: 0 or 1)', fontsize=14)
        
        plt.tight_layout()
        
        safe_name = model.replace('.', '_').replace('-', '_')
        output_path = f'{output_dir}/confusion_{safe_name}.png'
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

    # Generate plots by model
    print("\nGenerating plots by model...")
    plot_by_model(stats)
    
    # Compute and plot accuracy
    print("\nComputing accuracy...")
    accuracy = compute_accuracy(df)
    print("\nAccuracy by model and N:")
    print(accuracy.to_string(index=False))
    plot_accuracy(accuracy)
    
    # Compute and plot confusion matrices
    print("\nGenerating confusion matrices...")
    confusion_data = compute_confusion_matrix(df)
    plot_confusion_matrices(confusion_data)

    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
