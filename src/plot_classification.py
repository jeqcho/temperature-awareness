"""Plot classification experiment results.

For each classifier model, create a figure with subplots by source model.
X-axis: N (number of warmup questions)
Y-axis: guessed temperature
Lines: actual temperature (0, 0.5, 1.0)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(filename: str = 'results_classification/classification_results.json') -> list[dict]:
    """Load classification results from JSON file."""
    with open(filename) as f:
        return json.load(f)


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert results to pandas DataFrame."""
    return pd.DataFrame(results)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and stderr for each group."""
    # Filter to valid results
    df_valid = df[df['guessed_temperature'].notna()].copy()
    
    # Group by classifier, source model, actual temp, and n_questions
    grouped = df_valid.groupby([
        'classifier_model', 
        'source_model', 
        'actual_temperature', 
        'n_questions'
    ])
    
    stats = grouped['guessed_temperature'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
    ]).reset_index()
    
    # Calculate standard error
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
    
    return stats


def plot_classifier_results(stats: pd.DataFrame, output_dir: str = 'plots/classification'):
    """Create one figure per classifier model with subplots by source model."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    classifier_models = sorted(stats['classifier_model'].unique())
    source_models = sorted(stats['source_model'].unique())
    actual_temps = sorted(stats['actual_temperature'].unique())
    
    # Colors for actual temperatures
    temp_colors = {0.0: '#2ecc71', 0.5: '#3498db', 1.0: '#e74c3c'}
    temp_labels = {0.0: 'Actual T=0.0', 0.5: 'Actual T=0.5', 1.0: 'Actual T=1.0'}
    
    for classifier in classifier_models:
        classifier_data = stats[stats['classifier_model'] == classifier]
        
        # Create figure with subplots (2x2 for 4 source models)
        n_sources = len(source_models)
        n_cols = 2
        n_rows = (n_sources + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
        axes = axes.flatten() if n_sources > 1 else [axes]
        
        for idx, source in enumerate(source_models):
            ax = axes[idx]
            source_data = classifier_data[classifier_data['source_model'] == source]
            
            # Order: T=1.0 first, then T=0.5, then T=0.0
            for actual_temp in [1.0, 0.5, 0.0]:
                temp_data = source_data[source_data['actual_temperature'] == actual_temp]
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
                    capsize=5,
                    linewidth=2.5,
                    markersize=10,
                )
                
                # Add horizontal line for actual temperature (reference)
                ax.axhline(y=actual_temp, color=temp_colors[actual_temp], 
                          linestyle='--', alpha=0.3, linewidth=2)
            
            ax.set_xlabel('Number of Warmup Questions (N)', fontsize=14)
            ax.set_ylabel('Guessed Temperature', fontsize=14)
            ax.set_title(f'Source: {source}', fontsize=16)
            ax.set_xticks([0, 1, 2, 4, 8, 16])
            ax.tick_params(axis='both', labelsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(source_models), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f'Classifier: {classifier}\nGuessed Temperature by Source Model', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = f'{output_dir}/classifier_{classifier.replace(".", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_combined_heatmap(stats: pd.DataFrame, output_dir: str = 'plots/classification'):
    """Create a heatmap showing classification accuracy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Compute accuracy: how close is the guess to the actual temperature?
    df = stats.copy()
    df['error'] = abs(df['mean'] - df['actual_temperature'])
    
    # Aggregate across n_questions for overall accuracy
    accuracy = df.groupby(['classifier_model', 'source_model', 'actual_temperature']).agg({
        'mean': 'mean',
        'error': 'mean'
    }).reset_index()
    
    print("\nClassification Summary:")
    print(accuracy.to_string(index=False))


def main():
    print("Loading classification results...")
    results = load_results()
    df = results_to_dataframe(results)
    
    print(f"Total results: {len(df)}")
    print(f"Classifier models: {df['classifier_model'].unique().tolist()}")
    print(f"Source models: {df['source_model'].unique().tolist()}")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_classifier_results(stats)
    plot_combined_heatmap(stats)
    
    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
