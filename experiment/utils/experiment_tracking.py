"""
Experiment tracking utilities for organizing and monitoring experiments.

This module provides tools to systematically track hyperparameters, metrics,
and artifacts across multiple experimental runs.
"""

import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class ExperimentTracker:
    """
    Track experiments with hyperparameters, metrics, and artifacts.

    Usage:
        tracker = ExperimentTracker('baseline_benchmark_SZH')
        tracker.log_hyperparameters({'model': 'moderntcn', 'lr': 1e-4})
        tracker.log_metrics({'MAE': 12.5, 'RMSE': 18.3})
        tracker.save_predictions(predictions, labels)
    """

    def __init__(self, experiment_name: str, save_dir: str = 'results/experiments/'):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            save_dir: Base directory for saving results
        """
        self.experiment_name = experiment_name
        self.save_dir = os.path.join(save_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.save_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics = {}
        self.hyperparameters = {}

        print(f"📊 Experiment Tracker Initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Save Dir: {self.run_dir}")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameters
        """
        self.hyperparameters.update(params)

        # Save to JSON file
        with open(os.path.join(self.run_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(self.hyperparameters, f, indent=4)

        print(f"✓ Logged {len(params)} hyperparameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics (MAE, RMSE, etc.).

        Args:
            metrics: Dictionary of metric values
            step: Optional step/epoch number for tracking progression
        """
        if step is not None:
            if step not in self.metrics:
                self.metrics[step] = {}
            self.metrics[step].update(metrics)
        else:
            self.metrics.update(metrics)

        # Save to JSON file
        with open(os.path.join(self.run_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)

        print(f"✓ Logged metrics: {', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])}")

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Log artifact (model checkpoint, plots, etc.).

        Args:
            artifact_path: Path to artifact file
            artifact_name: Optional custom name for artifact
        """
        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)

        import shutil
        dest_path = os.path.join(self.run_dir, artifact_name)
        shutil.copy(artifact_path, dest_path)

        print(f"✓ Logged artifact: {artifact_name}")

    def save_predictions(self, predictions: np.ndarray, labels: np.ndarray, split: str = 'test'):
        """
        Save predictions and labels.

        Args:
            predictions: Model predictions array
            labels: Ground truth labels array
            split: Data split name (train/val/test)
        """
        pred_path = os.path.join(self.run_dir, f'{split}_predictions.npy')
        label_path = os.path.join(self.run_dir, f'{split}_labels.npy')

        np.save(pred_path, predictions)
        np.save(label_path, labels)

        print(f"✓ Saved {split} predictions and labels")

    def save_model(self, model_state_dict: Dict, model_name: str = 'model.pth'):
        """
        Save model checkpoint.

        Args:
            model_state_dict: Model state dictionary
            model_name: Name for saved model file
        """
        import torch

        model_path = os.path.join(self.run_dir, model_name)
        torch.save(model_state_dict, model_path)

        print(f"✓ Saved model checkpoint: {model_name}")

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create summary table aggregating all experiments.

        Returns:
            DataFrame with all experiment results
        """
        summary_file = os.path.join(self.save_dir, 'summary.csv')

        # Load existing summary if it exists
        if os.path.exists(summary_file):
            summary = pd.read_csv(summary_file)
        else:
            summary = pd.DataFrame()

        # Add current run
        run_summary = {
            'run_id': self.run_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **self.hyperparameters,
            **self.metrics
        }

        summary = pd.concat([summary, pd.DataFrame([run_summary])], ignore_index=True)
        summary.to_csv(summary_file, index=False)

        print(f"✓ Updated summary table: {summary_file}")

        return summary

    def get_run_dir(self) -> str:
        """Get the current run directory path."""
        return self.run_dir

    def print_summary(self):
        """Print a summary of the current experiment."""
        print("\n" + "="*60)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("="*60)
        print(f"Run ID: {self.run_id}")
        print(f"\nHyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print(f"\nMetrics:")
        if isinstance(self.metrics, dict) and any(isinstance(v, dict) for v in self.metrics.values()):
            # Step-based metrics
            for step, metrics in sorted(self.metrics.items()):
                if isinstance(metrics, dict):
                    print(f"  Step {step}:")
                    for key, value in metrics.items():
                        print(f"    {key}: {value:.4f}")
        else:
            # Single metrics
            for key, value in self.metrics.items():
                print(f"  {key}: {value:.4f}")
        print("="*60 + "\n")


def load_experiment_results(experiment_name: str, save_dir: str = 'results/experiments/') -> pd.DataFrame:
    """
    Load all results from a specific experiment.

    Args:
        experiment_name: Name of experiment to load
        save_dir: Base directory where experiments are saved

    Returns:
        DataFrame with all experiment runs
    """
    summary_file = os.path.join(save_dir, experiment_name, 'summary.csv')

    if os.path.exists(summary_file):
        return pd.read_csv(summary_file)
    else:
        print(f"⚠️  No results found for experiment: {experiment_name}")
        return pd.DataFrame()


def compare_experiments(experiment_names: list, metric: str = 'MAE', save_dir: str = 'results/experiments/') -> pd.DataFrame:
    """
    Compare multiple experiments on a specific metric.

    Args:
        experiment_names: List of experiment names to compare
        metric: Metric to compare (default: MAE)
        save_dir: Base directory where experiments are saved

    Returns:
        Comparison DataFrame
    """
    results = []

    for exp_name in experiment_names:
        df = load_experiment_results(exp_name, save_dir)
        if not df.empty and metric in df.columns:
            results.append({
                'experiment': exp_name,
                'best_' + metric: df[metric].min(),
                'mean_' + metric: df[metric].mean(),
                'std_' + metric: df[metric].std(),
                'num_runs': len(df)
            })

    return pd.DataFrame(results)