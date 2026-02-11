"""Statistical significance testing for benchmark results.

Performs rigorous statistical tests:
- Paired t-tests between models
- Wilcoxon signed-rank test (non-parametric)
- Effect size (Cohen's d)
- Confidence intervals
- Multiple testing correction (Bonferroni)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalAnalyzer:
    """Perform statistical analysis on benchmark results."""
    
    def __init__(self, results_path: str):
        """Initialize with results CSV.
        
        Args:
            results_path: Path to benchmark results CSV
        """
        self.df = pd.read_csv(results_path)
        self.significance_level = 0.05
    
    def paired_t_test(
        self,
        model1: str,
        model2: str,
        metric: str = 'r2_mean'
    ) -> Dict:
        """Perform paired t-test between two models.
        
        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare
        
        Returns:
            Dictionary with test results
        """
        scores1 = self.df[self.df['model'] == model1][metric].values
        scores2 = self.df[self.df['model'] == model2][metric].values
        
        if len(scores1) == 0 or len(scores2) == 0:
            return {'error': 'Model not found'}
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Effect size (Cohen's d)
        diff = scores1 - scores2
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # Confidence interval
        ci = stats.t.interval(
            0.95,
            len(diff) - 1,
            loc=np.mean(diff),
            scale=stats.sem(diff)
        )
        
        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(cohens_d),
            'mean_diff': np.mean(diff),
            'ci_95': ci
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
        
        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def wilcoxon_test(
        self,
        model1: str,
        model2: str,
        metric: str = 'r2_mean'
    ) -> Dict:
        """Non-parametric Wilcoxon signed-rank test.
        
        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare
        
        Returns:
            Dictionary with test results
        """
        scores1 = self.df[self.df['model'] == model1][metric].values
        scores2 = self.df[self.df['model'] == model2][metric].values
        
        if len(scores1) == 0 or len(scores2) == 0:
            return {'error': 'Model not found'}
        
        # Wilcoxon test
        stat, p_value = stats.wilcoxon(scores1, scores2)
        
        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'test': 'wilcoxon',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level
        }
    
    def pairwise_comparison(
        self,
        models: List[str],
        metric: str = 'r2_mean',
        method: str = 'ttest'
    ) -> pd.DataFrame:
        """Perform pairwise comparisons between all models.
        
        Args:
            models: List of model names
            metric: Metric to compare
            method: 'ttest' or 'wilcoxon'
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                if method == 'ttest':
                    result = self.paired_t_test(model1, model2, metric)
                else:
                    result = self.wilcoxon_test(model1, model2, metric)
                
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Bonferroni correction
        if 'p_value' in df.columns:
            n_comparisons = len(results)
            df['p_value_corrected'] = df['p_value'] * n_comparisons
            df['p_value_corrected'] = df['p_value_corrected'].clip(upper=1.0)
            df['significant_corrected'] = df['p_value_corrected'] < self.significance_level
        
        return df
    
    def generate_significance_matrix(
        self,
        models: List[str],
        metric: str = 'r2_mean'
    ) -> np.ndarray:
        """Generate matrix of p-values for all model pairs.
        
        Args:
            models: List of model names
            metric: Metric to compare
        
        Returns:
            Matrix of p-values
        """
        n = len(models)
        matrix = np.ones((n, n))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    result = self.paired_t_test(model1, model2, metric)
                    if 'p_value' in result:
                        matrix[i, j] = result['p_value']
        
        return matrix
    
    def plot_significance_matrix(
        self,
        models: List[str],
        metric: str = 'r2_mean',
        save_path: str = None
    ):
        """Visualize statistical significance matrix.
        
        Args:
            models: List of model names
            metric: Metric to compare
            save_path: Path to save figure
        """
        matrix = self.generate_significance_matrix(models, metric)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for diagonal
        mask = np.eye(len(models), dtype=bool)
        
        # Plot heatmap
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=0.1,
            xticklabels=models,
            yticklabels=models,
            cbar_kws={'label': 'p-value'},
            ax=ax
        )
        
        ax.set_title(f'Statistical Significance Matrix ({metric})', 
                    fontsize=16, fontweight='bold')
        
        # Add significance threshold line
        ax.axhline(y=0, color='red', linewidth=2, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved significance matrix: {save_path}")
        
        plt.close()
    
    def critical_difference_diagram(
        self,
        models: List[str],
        metric: str = 'r2_mean',
        save_path: str = None
    ):
        """Generate critical difference diagram (Demšar, 2006).
        
        Args:
            models: List of model names
            metric: Metric to compare
            save_path: Path to save figure
        """
        # Get average ranks
        ranks = []
        for dataset in self.df['dataset'].unique():
            subset = self.df[self.df['dataset'] == dataset]
            subset = subset[subset['model'].isin(models)]
            subset = subset.sort_values(metric, ascending=False)
            
            model_ranks = {}
            for rank, (_, row) in enumerate(subset.iterrows(), 1):
                model_ranks[row['model']] = rank
            
            ranks.append(model_ranks)
        
        # Average ranks
        avg_ranks = {}
        for model in models:
            avg_ranks[model] = np.mean([r.get(model, len(models)) for r in ranks])
        
        # Sort by rank
        sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        # Critical difference (Nemenyi test)
        n_datasets = len(ranks)
        n_models = len(models)
        q_alpha = 2.576  # For alpha=0.05, k=10 (approximate)
        cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6 * n_datasets))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        y_pos = np.arange(len(sorted_models))
        ranks_values = [r for _, r in sorted_models]
        labels = [m for m, _ in sorted_models]
        
        ax.barh(y_pos, ranks_values, color='skyblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Average Rank', fontsize=14)
        ax.set_title('Critical Difference Diagram', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        
        # Add CD line
        best_rank = min(ranks_values)
        ax.axvline(x=best_rank + cd, color='red', linestyle='--', 
                  linewidth=2, label=f'CD = {cd:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved CD diagram: {save_path}")
        
        plt.close()


def main():
    """Run statistical analysis."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python statistical_analysis.py <results.csv>")
        return
    
    results_path = sys.argv[1]
    
    print("\n" + "="*80)
    print(" "*20 + "STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    analyzer = StatisticalAnalyzer(results_path)
    
    # Get all models
    models = analyzer.df['model'].unique().tolist()
    print(f"Analyzing {len(models)} models: {', '.join(models)}\n")
    
    # Pairwise comparisons
    print("Pairwise t-tests with Bonferroni correction:")
    print("-" * 80)
    
    comparisons = analyzer.pairwise_comparison(models, metric='r2_mean')
    print(comparisons[['model1', 'model2', 'p_value', 'p_value_corrected', 
                       'significant_corrected', 'cohens_d', 'effect_size']].to_string(index=False))
    
    # Save results
    output_dir = Path(results_path).parent
    comparisons.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print(f"\n✓ Saved statistical tests: {output_dir / 'statistical_tests.csv'}")
    
    # Generate visualizations
    analyzer.plot_significance_matrix(
        models,
        save_path=str(output_dir / 'significance_matrix.png')
    )
    
    analyzer.critical_difference_diagram(
        models,
        save_path=str(output_dir / 'critical_difference.png')
    )
    
    print("\n" + "="*80)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
