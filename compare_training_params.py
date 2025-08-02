import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

def compare_training_runs(log_dir="training_logs"):
    """Compare different training runs to show parameter impact"""
    
    # Find all DQN episode files
    episode_files = sorted(glob(os.path.join(log_dir, "dqn_episodes_*.csv")))
    
    if len(episode_files) < 2:
        print("Need at least 2 training runs to compare. Run training with different parameters first.")
        return
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Set Comparison: Default vs Fast Learning', fontsize=16)
    
    # Colors for different runs
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    labels = ['Default (slow epsilon)', 'Fast Learning', 'Stable', 'Exploration']
    
    # Load and plot each run
    for idx, file in enumerate(episode_files[-2:]):  # Compare last 2 runs
        df = pd.read_csv(file)
        color = colors[idx]
        label = labels[idx] if idx < len(labels) else f"Run {idx+1}"
        
        # 1. Score progression
        ax = axes[0, 0]
        ax.plot(df['episode'], df['score'], alpha=0.3, color=color)
        moving_avg = df['score'].rolling(window=50, min_periods=1).mean()
        ax.plot(df['episode'], moving_avg, linewidth=2, color=color, label=label)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Score Progression Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Epsilon decay comparison
        ax = axes[0, 1]
        ax.plot(df['episode'], df['epsilon'], linewidth=2, color=color, label=label)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon) Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Average loss
        ax = axes[1, 0]
        valid_losses = df[df['avg_loss'] > 0]['avg_loss']
        if len(valid_losses) > 0:
            loss_moving_avg = valid_losses.rolling(window=20, min_periods=1).mean()
            ax.plot(df.index[df['avg_loss'] > 0][:len(loss_moving_avg)], 
                   loss_moving_avg, linewidth=2, color=color, label=label)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Loss')
        ax.set_title('Training Loss Comparison')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Performance metrics table
        ax = axes[1, 1]
        if idx == 0:
            ax.axis('off')
            metrics_text = "Performance Metrics Comparison\n\n"
            metrics_text += f"{'Metric':<25} {'Default':<15} {'Fast Learning':<15}\n"
            metrics_text += "-" * 60 + "\n"
        
        # Calculate metrics
        final_100_scores = df['score'].tail(100).mean() if len(df) >= 100 else df['score'].mean()
        max_score = df['score'].max()
        episodes_to_50 = len(df[df['score'].rolling(50).mean() < 50]) if len(df) >= 50 else "N/A"
        final_epsilon = df['epsilon'].iloc[-1]
        
        if idx == 0:
            default_metrics = {
                'avg_final_100': final_100_scores,
                'max_score': max_score,
                'episodes_to_50': episodes_to_50,
                'final_epsilon': final_epsilon
            }
        else:
            metrics_text += f"{'Avg Score (last 100)':<25} {default_metrics['avg_final_100']:<15.2f} {final_100_scores:<15.2f}\n"
            metrics_text += f"{'Max Score Achieved':<25} {default_metrics['max_score']:<15.0f} {max_score:<15.0f}\n"
            metrics_text += f"{'Episodes to avg 50':<25} {str(default_metrics['episodes_to_50']):<15} {str(episodes_to_50):<15}\n"
            metrics_text += f"{'Final Epsilon':<25} {default_metrics['final_epsilon']:<15.4f} {final_epsilon:<15.4f}\n"
            
            # Performance improvement
            improvement = ((final_100_scores - default_metrics['avg_final_100']) / default_metrics['avg_final_100']) * 100
            metrics_text += f"\n{'Performance Change':<25} {improvement:+.1f}%"
            
            ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', 
                   verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save the comparison
    output_file = os.path.join("training_analysis_comparison", "parameter_comparison.png")
    os.makedirs("training_analysis_comparison", exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison saved to: {output_file}")
    
    # Also create a detailed parameter comparison table
    create_parameter_comparison_table(episode_files[-2:])

def create_parameter_comparison_table(episode_files):
    """Create a detailed table comparing parameters and results"""
    
    comparison_data = []
    
    for idx, file in enumerate(episode_files):
        # Load episode data
        df = pd.read_csv(file)
        
        # Load parameter data if exists
        timestamp = file.split('_')[-1].replace('.csv', '')
        param_file = file.replace('episodes', 'params')
        
        if os.path.exists(param_file):
            param_df = pd.read_csv(param_file)
            initial_params = param_df.iloc[0] if len(param_df) > 0 else {}
            final_params = param_df.iloc[-1] if len(param_df) > 0 else {}
        else:
            initial_params = final_params = {}
        
        # Calculate key metrics
        metrics = {
            'Run': f"Run {idx+1}",
            'Total Episodes': len(df),
            'Initial Learning Rate': initial_params.get('learning_rate', 'N/A'),
            'Gamma': initial_params.get('gamma', 'N/A'),
            'Initial Epsilon': initial_params.get('epsilon', 'N/A'),
            'Final Epsilon': final_params.get('epsilon', df['epsilon'].iloc[-1] if 'epsilon' in df else 'N/A'),
            'Epsilon Decay': initial_params.get('epsilon_decay', 'N/A'),
            'Batch Size': initial_params.get('batch_size', 'N/A'),
            'Memory Size': initial_params.get('memory_size', 'N/A'),
            'Update Frequency': initial_params.get('update_freq', 'N/A'),
            'Avg Score (all)': df['score'].mean(),
            'Avg Score (last 100)': df['score'].tail(100).mean() if len(df) >= 100 else df['score'].mean(),
            'Max Score': df['score'].max(),
            'Min Score': df['score'].min(),
            'Score Std Dev': df['score'].std(),
            'Avg Loss': df[df['avg_loss'] > 0]['avg_loss'].mean() if 'avg_loss' in df else 'N/A',
            'Final Loss': df[df['avg_loss'] > 0]['avg_loss'].iloc[-1] if 'avg_loss' in df and len(df[df['avg_loss'] > 0]) > 0 else 'N/A',
            'Avg Survival Time': df['survival_time'].mean() if 'survival_time' in df else 'N/A',
            'Total Asteroids': df['asteroids_destroyed'].sum() if 'asteroids_destroyed' in df else 'N/A',
            'Avg Accuracy': df['accuracy'].mean() if 'accuracy' in df else 'N/A'
        }
        
        comparison_data.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    output_file = os.path.join("training_analysis_comparison", "parameter_comparison_table.csv")
    comparison_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nDetailed Parameter Comparison:")
    print("="*80)
    
    # Print key differences
    if len(comparison_data) >= 2:
        print("\nKey Parameter Differences:")
        print(f"Learning Rate: {comparison_data[0]['Initial Learning Rate']} -> {comparison_data[1]['Initial Learning Rate']}")
        print(f"Epsilon Decay: {comparison_data[0]['Epsilon Decay']} -> {comparison_data[1]['Epsilon Decay']}")
        print(f"Batch Size: {comparison_data[0]['Batch Size']} -> {comparison_data[1]['Batch Size']}")
        print(f"Memory Size: {comparison_data[0]['Memory Size']} -> {comparison_data[1]['Memory Size']}")
        
        print("\nPerformance Impact:")
        score_change = comparison_data[1]['Avg Score (last 100)'] - comparison_data[0]['Avg Score (last 100)']
        score_pct = (score_change / comparison_data[0]['Avg Score (last 100)']) * 100
        print(f"Average Score Change: {score_change:+.2f} ({score_pct:+.1f}%)")
        
        if isinstance(comparison_data[0]['Final Epsilon'], (int, float)) and isinstance(comparison_data[1]['Final Epsilon'], (int, float)):
            epsilon_diff = comparison_data[1]['Final Epsilon'] - comparison_data[0]['Final Epsilon']
            print(f"Final Epsilon Difference: {epsilon_diff:+.4f}")
    
    print(f"\nFull comparison table saved to: {output_file}")

if __name__ == "__main__":
    print("Asteroids AI Parameter Comparison Analysis")
    print("="*60)
    compare_training_runs()