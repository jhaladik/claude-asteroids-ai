import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from glob import glob
# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

class TrainingAnalyzer:
    """Analyze and visualize AI training logs"""
    
    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        self.episode_data = None
        self.step_data = None
        self.param_data = None
        self.task_data = None
        self.summary_data = None
        
    def load_latest_logs(self, agent_type="dqn"):
        """Load the most recent log files for a given agent type"""
        # Find latest files
        episode_files = sorted(glob(os.path.join(self.log_dir, f"{agent_type}_episodes_*.csv")))
        if not episode_files:
            print(f"No episode logs found for {agent_type}")
            return False
            
        latest_episode = episode_files[-1]
        timestamp = latest_episode.split('_')[-1].replace('.csv', '')
        
        # Load episode data
        self.episode_data = pd.read_csv(latest_episode)
        print(f"Loaded episode data: {len(self.episode_data)} episodes")
        
        # Load step data if exists
        step_file = os.path.join(self.log_dir, f"{agent_type}_steps_{timestamp}.csv")
        if os.path.exists(step_file):
            self.step_data = pd.read_csv(step_file)
            print(f"Loaded step data: {len(self.step_data)} steps")
        
        # Load parameter data
        param_file = os.path.join(self.log_dir, f"{agent_type}_params_{timestamp}.csv")
        if os.path.exists(param_file):
            self.param_data = pd.read_csv(param_file)
            print(f"Loaded parameter data")
        
        # Load task data for meta-learning
        if agent_type == "meta":
            task_file = os.path.join(self.log_dir, f"{agent_type}_tasks_{timestamp}.csv")
            if os.path.exists(task_file):
                self.task_data = pd.read_csv(task_file)
                print(f"Loaded task data")
        
        # Load summary
        summary_file = os.path.join(self.log_dir, f"{agent_type}_summary_{timestamp}.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                self.summary_data = json.load(f)
                print(f"Loaded summary data")
        
        return True
    
    def plot_learning_curves(self, save_path=None):
        """Plot comprehensive learning curves"""
        if self.episode_data is None:
            print("No episode data loaded")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('AI Training Analysis', fontsize=16)
        
        # 1. Score progression
        ax = axes[0, 0]
        ax.plot(self.episode_data['episode'], self.episode_data['score'], alpha=0.3, label='Score')
        # Calculate moving average
        window = 50
        moving_avg = self.episode_data['score'].rolling(window=window, min_periods=1).mean()
        ax.plot(self.episode_data['episode'], moving_avg, linewidth=2, label=f'{window}-episode avg')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Score Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss progression
        ax = axes[0, 1]
        if 'avg_loss' in self.episode_data.columns:
            valid_losses = self.episode_data[self.episode_data['avg_loss'] > 0]['avg_loss']
            if len(valid_losses) > 0:
                ax.plot(self.episode_data.index[self.episode_data['avg_loss'] > 0], 
                       valid_losses, alpha=0.5)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Average Loss')
                ax.set_title('Training Loss')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
        
        # 3. Q-value progression
        ax = axes[1, 0]
        if 'avg_q_value' in self.episode_data.columns:
            valid_q = self.episode_data[self.episode_data['avg_q_value'] != 0]['avg_q_value']
            if len(valid_q) > 0:
                ax.plot(self.episode_data.index[self.episode_data['avg_q_value'] != 0], 
                       valid_q)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Average Q-value')
                ax.set_title('Q-value Evolution')
                ax.grid(True, alpha=0.3)
        
        # 4. Epsilon decay
        ax = axes[1, 1]
        ax.plot(self.episode_data['episode'], self.episode_data['epsilon'])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon) Decay')
        ax.grid(True, alpha=0.3)
        
        # 5. Accuracy and survival
        ax = axes[2, 0]
        if 'accuracy' in self.episode_data.columns:
            ax.plot(self.episode_data['episode'], self.episode_data['accuracy'], 
                   label='Shooting Accuracy', alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(self.episode_data['episode'], self.episode_data['survival_time'], 
                color='orange', label='Survival Time', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Survival Time')
        ax.set_title('Performance Metrics')
        ax.grid(True, alpha=0.3)
        
        # 6. Action diversity
        ax = axes[2, 1]
        if 'action_entropy' in self.episode_data.columns:
            ax.plot(self.episode_data['episode'], self.episode_data['action_entropy'])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Action Entropy')
            ax.set_title('Action Diversity')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def analyze_performance_degradation(self):
        """Analyze where and why performance degrades"""
        if self.episode_data is None:
            print("No episode data loaded")
            return
        
        # Split data into segments
        n_segments = 10
        segment_size = len(self.episode_data) // n_segments
        
        print("\nPerformance Analysis by Training Phase:")
        print("="*60)
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(self.episode_data)
            segment = self.episode_data.iloc[start_idx:end_idx]
            
            avg_score = segment['score'].mean()
            std_score = segment['score'].std()
            avg_epsilon = segment['epsilon'].mean()
            avg_loss = segment[segment['avg_loss'] > 0]['avg_loss'].mean() if 'avg_loss' in segment else 0
            
            print(f"\nPhase {i+1} (Episodes {start_idx}-{end_idx}):")
            print(f"  Avg Score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"  Epsilon: {avg_epsilon:.4f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            
            # Check for performance drop
            if i > 0:
                prev_segment = self.episode_data.iloc[(i-1)*segment_size:i*segment_size]
                prev_avg = prev_segment['score'].mean()
                if avg_score < prev_avg * 0.8:  # 20% drop
                    print(f"  WARNING: Performance DROP detected! ({prev_avg:.1f} -> {avg_score:.1f})")
    
    def suggest_parameter_improvements(self):
        """Suggest parameter improvements based on analysis"""
        if self.episode_data is None or self.param_data is None:
            print("Insufficient data for parameter suggestions")
            return
        
        print("\nParameter Optimization Suggestions:")
        print("="*60)
        
        # Analyze epsilon decay
        final_epsilon = self.episode_data.iloc[-1]['epsilon']
        if final_epsilon > 0.05:
            print("\n1. EPSILON DECAY TOO SLOW")
            print(f"   Current final epsilon: {final_epsilon:.4f}")
            print("   Suggestion: Increase epsilon_decay to reach 0.01 faster")
            print("   Recommended: epsilon_decay = 0.995 (reaches 0.01 in ~900 episodes)")
        
        # Analyze learning stability
        score_variance = self.episode_data['score'].rolling(window=50).std().mean()
        if score_variance > 20:
            print("\n2. HIGH SCORE VARIANCE")
            print(f"   Average variance: {score_variance:.2f}")
            print("   Suggestions:")
            print("   - Decrease learning_rate to 0.0005")
            print("   - Increase batch_size to 64")
            print("   - Add target network with update_freq = 100")
        
        # Analyze Q-value explosion
        if 'avg_q_value' in self.episode_data.columns:
            q_values = self.episode_data[self.episode_data['avg_q_value'] != 0]['avg_q_value']
            if len(q_values) > 100:
                q_growth = q_values.iloc[-50:].mean() / q_values.iloc[:50].mean()
                if q_growth > 10:
                    print("\n3. Q-VALUE EXPLOSION DETECTED")
                    print(f"   Q-value growth factor: {q_growth:.2f}x")
                    print("   Suggestions:")
                    print("   - Reduce gamma to 0.9")
                    print("   - Add gradient clipping")
                    print("   - Implement Double DQN")
        
        # Memory usage
        if self.param_data is not None and 'memory_size' in self.param_data.columns:
            max_memory = self.param_data['memory_size'].max()
            if max_memory < 5000:
                print("\n4. INSUFFICIENT EXPERIENCE REPLAY")
                print(f"   Max memory used: {max_memory}")
                print("   Suggestion: Increase memory buffer to 20000")
    
    def plot_task_analysis(self, save_path=None):
        """Plot task-specific analysis for meta-learning"""
        if self.task_data is None:
            print("No task data available")
            return
        
        # Aggregate task statistics
        task_stats = self.task_data.groupby('task').agg({
            'usage_count': 'sum',
            'avg_reward': 'mean',
            'success_rate': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Meta-Learning Task Analysis', fontsize=16)
        
        # 1. Task usage distribution
        ax = axes[0, 0]
        task_stats.plot(x='task', y='usage_count', kind='bar', ax=ax, legend=False)
        ax.set_title('Task Usage Distribution')
        ax.set_xlabel('Task')
        ax.set_ylabel('Usage Count')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Task rewards
        ax = axes[0, 1]
        task_stats.plot(x='task', y='avg_reward', kind='bar', ax=ax, legend=False, color='orange')
        ax.set_title('Average Reward by Task')
        ax.set_xlabel('Task')
        ax.set_ylabel('Average Reward')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Task performance over time
        ax = axes[1, 0]
        for task in self.task_data['task'].unique():
            task_episodes = self.task_data[self.task_data['task'] == task]
            ax.plot(task_episodes['episode'], 
                   task_episodes['avg_reward'].rolling(window=10).mean(), 
                   label=task, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Task Reward (10-ep avg)')
        ax.set_title('Task Performance Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Success rates
        ax = axes[1, 1]
        task_stats.plot(x='task', y='success_rate', kind='bar', ax=ax, legend=False, color='green')
        ax.set_title('Success Rate by Task')
        ax.set_xlabel('Task')
        ax.set_ylabel('Success Rate')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def main():
    """Main analysis function"""
    analyzer = TrainingAnalyzer()
    
    print("AI Training Log Analyzer")
    print("="*60)
    
    # Check for available logs
    if not os.path.exists("training_logs"):
        print("No training_logs directory found. Please train an AI first.")
        return
    
    # List available agent types
    log_files = os.listdir("training_logs")
    agent_types = set()
    for file in log_files:
        if file.endswith('.csv'):
            agent_type = file.split('_')[0]
            agent_types.add(agent_type)
    
    if not agent_types:
        print("No log files found in training_logs directory.")
        return
    
    print(f"Found logs for agent types: {', '.join(agent_types)}")
    
    # Analyze each agent type
    for agent_type in agent_types:
        print(f"\n\nAnalyzing {agent_type.upper()} Agent")
        print("-"*60)
        
        if analyzer.load_latest_logs(agent_type):
            # Create output directory
            output_dir = f"training_analysis_{agent_type}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate plots
            analyzer.plot_learning_curves(
                save_path=os.path.join(output_dir, "learning_curves.png")
            )
            
            # Performance analysis
            analyzer.analyze_performance_degradation()
            
            # Parameter suggestions
            analyzer.suggest_parameter_improvements()
            
            # Task analysis for meta-learning
            if agent_type == "meta":
                analyzer.plot_task_analysis(
                    save_path=os.path.join(output_dir, "task_analysis.png")
                )
            
            print(f"\nAnalysis saved to {output_dir}/")

if __name__ == "__main__":
    main()