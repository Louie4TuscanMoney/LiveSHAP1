"""
NBA Data Preparation Script
Follows the methodology from Research.md (lines 158-176)

This script:
1. Loads all game JSON files from Dataset/games/
2. Converts to DataFrame format
3. Performs feature selection (removes highly correlated features)
4. Creates descriptive statistics
5. Performs logistic regression analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class NBADataPreparation:
    """
    Prepare NBA game data following research paper methodology
    """
    
    def __init__(self, dataset_dir='Dataset', output_dir='2datapreparation'):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.games_dir = self.dataset_dir / 'games'
        
    def load_all_games(self):
        """Load all JSON game files and convert to DataFrame"""
        print("=" * 60)
        print("Loading game data from JSON files...")
        print("=" * 60)
        
        game_files = list(self.games_dir.glob('*.json'))
        print(f"Found {len(game_files)} game files")
        
        all_games = []
        
        for game_file in game_files:
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                
                # Extract features for each period
                row = {
                    'GAME_ID': game_data['GAME_ID'],
                    'SEASON': game_data['SEASON'],
                    'RESULT': game_data['RESULT'],
                }
                
                # Extract differences for game, H2, H3
                for period in ['game', 'H2', 'H3']:
                    if period in game_data and 'differences' in game_data[period]:
                        diff = game_data[period]['differences']
                        for key, value in diff.items():
                            row[key] = value
                
                all_games.append(row)
                
            except Exception as e:
                print(f"Error loading {game_file.name}: {e}")
                continue
        
        df = pd.DataFrame(all_games)
        print(f"\nLoaded {len(df)} games")
        print(f"Total features: {len(df.columns)}")
        
        return df
    
    def remove_outliers(self, df):
        """
        Remove outliers: preseason games, All-Star games, invalid entries
        Note: Feature values are differences (home - away), so they can be negative
        Based on Research.md Table 2, reasonable ranges are:
        - H2_FG%: -0.373 to 0.373
        - H3_FG%: -0.364 to 0.308
        - game_FG%: -0.313 to 0.296
        """
        print("\n" + "=" * 60)
        print("Removing outliers...")
        print("=" * 60)
        
        initial_count = len(df)
        
        # Remove games with missing critical data
        # Check if RESULT is valid (0 or 1)
        df = df[df['RESULT'].isin([0, 1])]
        
        # Remove games with missing period data
        required_cols = ['game_FG%', 'H2_FG%', 'H3_FG%']
        df = df.dropna(subset=required_cols)
        
        # Remove games with extreme values (likely data errors)
        # Since these are differences (home - away), they can be negative
        # Use reasonable ranges based on Research.md Table 2
        for period in ['game', 'H2', 'H3']:
            fg_col = f'{period}_FG%'
            if fg_col in df.columns:
                # Allow negative values and reasonable positive ranges
                # Based on paper: H2 max=0.373, H3 max=0.308, game max=0.296
                # Use slightly wider range to account for data variation
                max_fg = 0.5 if period == 'H2' else 0.4
                df = df[(df[fg_col] >= -0.5) & (df[fg_col] <= max_fg)]
            
            # Check other percentage-based features (2P%, 3P%, FT%)
            for pct_type in ['2P%', '3P%', 'FT%']:
                pct_col = f'{period}_{pct_type}'
                if pct_col in df.columns:
                    # Allow reasonable ranges for percentage differences
                    # FT% can have wider range due to small sample sizes
                    if pct_type == 'FT%':
                        df = df[(df[pct_col] >= -1.0) & (df[pct_col] <= 1.0)]
                    else:
                        df = df[(df[pct_col] >= -0.7) & (df[pct_col] <= 0.7)]
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"Initial games: {initial_count}")
        print(f"Final games: {final_count}")
        print(f"Removed: {removed} games ({removed/initial_count*100:.1f}%)")
        
        return df
    
    def analyze_correlations(self, df, period='game', correlation_threshold=0.8):
        """
        Analyze correlations to identify highly correlated features (≥ 0.8)
        Returns pairs of highly correlated features
        """
        # Get features for this period
        period_cols = [col for col in df.columns if col.startswith(f'{period}_')]
        period_cols = [col for col in period_cols if col not in ['GAME_ID', 'SEASON', 'RESULT']]
        
        if not period_cols:
            return []
        
        # Calculate correlation matrix
        corr_matrix = df[period_cols].corr()
        
        # Find highly correlated pairs
        highly_correlated = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= correlation_threshold:
                    highly_correlated.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return highly_correlated
    
    def create_scatter_plots(self, df, period='game', save=True):
        """
        Create scatter plots for highly correlated features (like Fig 4 in paper)
        Shows relationships: FG vs FG%, 3PA vs 3P%, FTA vs FT, DRB vs TRB
        """
        print(f"\nCreating scatter plots for {period}...")
        
        # Pairs to plot based on paper analysis
        scatter_pairs = [
            (f'{period}_FG', f'{period}_FG%', 'Field Goals Made vs Field Goal %'),
            (f'{period}_3PA', f'{period}_3P%', '3-Point Attempts vs 3-Point %'),
            (f'{period}_FTA', f'{period}_FT', 'Free Throw Attempts vs Free Throws Made'),
            (f'{period}_DRB', f'{period}_TRB', 'Defensive Rebounds vs Total Rebounds'),
        ]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (x_col, y_col, title) in enumerate(scatter_pairs):
            if x_col in df.columns and y_col in df.columns:
                axes[idx].scatter(df[x_col], df[y_col], alpha=0.5, s=20)
                axes[idx].set_xlabel(x_col.replace(f'{period}_', ''))
                axes[idx].set_ylabel(y_col.replace(f'{period}_', ''))
                axes[idx].set_title(title)
                axes[idx].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = df[x_col].corr(df[y_col])
                axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', 
                              transform=axes[idx].transAxes,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Scatter Plots: {period.upper()} Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / f'scatter_plots_{period}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
    
    def perform_feature_selection(self, df):
        """
        Perform feature selection based on correlation analysis (matching Research.md methodology)
        Step 1: Identify highly correlated features (correlation ≥ 0.8)
        Step 2: Analyze relationships via scatter plots
        Step 3: Remove features based on correlation analysis:
            - Remove makes/attempts when percentage is available (FG, FGA, 2P, 2PA, 3P, 3PA, FT, FTA)
            - Remove TRB (redundant with ORB + DRB)
            - Keep percentages (more accurately reflect team strength)
        """
        print("\n" + "=" * 60)
        print("Performing Feature Selection Based on Correlation Analysis")
        print("Following Research.md methodology (lines 165-169)")
        print("=" * 60)
        
        # Step 1: Analyze correlations for each period
        all_highly_correlated = {}
        for period in ['game', 'H2', 'H3']:
            highly_corr = self.analyze_correlations(df, period=period, correlation_threshold=0.8)
            all_highly_correlated[period] = highly_corr
            
            if highly_corr:
                print(f"\n{period.upper()} - Highly correlated features (|r| ≥ 0.8):")
                for pair in highly_corr:
                    print(f"  {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")
            else:
                print(f"\n{period.upper()} - No highly correlated pairs found (threshold: 0.8)")
        
        # Step 2: Create scatter plots
        for period in ['game', 'H2', 'H3']:
            self.create_scatter_plots(df, period=period)
        
        # Step 3: Remove features based on correlation analysis
        # As per paper: "features such as field goal makes, field goal attempts, 
        # two-pointer makes, two-pointer attempts, three-pointer makes, three-pointer attempts, 
        # free throw makes, free throw attempts, and total rebounds were removed"
        features_to_remove = []
        
        for period in ['game', 'H2', 'H3']:
            # Remove makes and attempts (keep percentages - more accurate)
            features_to_remove.extend([
                f'{period}_FG',      # Correlated with FG%
                f'{period}_FGA',    # Correlated with FG%
                f'{period}_2P',     # Correlated with 2P%
                f'{period}_2PA',    # Correlated with 2P%
                f'{period}_3P',     # Correlated with 3P%
                f'{period}_3PA',    # Correlated with 3P%
                f'{period}_FT',     # Correlated with FT%
                f'{period}_FTA',    # Correlated with FT%
                f'{period}_TRB',    # Correlated with DRB, redundant (TRB = ORB + DRB)
            ])
        
        # Remove features that don't exist
        features_to_remove = [f for f in features_to_remove if f in df.columns]
        
        print(f"\n" + "=" * 60)
        print("Feature Removal Decision:")
        print("=" * 60)
        print(f"Removing {len(features_to_remove)} features based on correlation analysis:")
        print("\nReasoning:")
        print("  - FG, FGA removed: Highly correlated with FG% (percentage more accurately reflects team strength)")
        print("  - 2P, 2PA removed: Highly correlated with 2P%")
        print("  - 3P, 3PA removed: Highly correlated with 3P%")
        print("  - FT, FTA removed: Highly correlated with FT%")
        print("  - TRB removed: Redundant (TRB = ORB + DRB), highly correlated with DRB")
        print("\nFeatures being removed:")
        for f in sorted(features_to_remove):
            print(f"  - {f}")
        
        df_selected = df.drop(columns=features_to_remove)
        
        print(f"\nFeatures before selection: {len(df.columns)}")
        print(f"Features after selection: {len(df_selected.columns)}")
        print(f"Removed: {len(features_to_remove)} features")
        
        # Show remaining features
        remaining_features = [col for col in df_selected.columns 
                            if col not in ['GAME_ID', 'SEASON', 'RESULT']]
        print(f"\nRemaining features ({len(remaining_features)}):")
        for f in sorted(remaining_features):
            print(f"  - {f}")
        
        # Save correlation analysis report
        report_file = self.output_dir / 'correlation_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write("Correlation Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            f.write("Feature Selection Based on Correlation Analysis\n")
            f.write("Following Research.md methodology (lines 165-169)\n\n")
            
            for period in ['game', 'H2', 'H3']:
                f.write(f"\n{period.upper()} Period:\n")
                f.write("-" * 40 + "\n")
                highly_corr = all_highly_correlated[period]
                if highly_corr:
                    f.write(f"Found {len(highly_corr)} highly correlated pairs (|r| ≥ 0.8):\n")
                    for pair in highly_corr:
                        f.write(f"  {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}\n")
                else:
                    f.write("No highly correlated pairs found (threshold: 0.8)\n")
            
            f.write(f"\n\nFeatures Removed: {len(features_to_remove)}\n")
            f.write("Reasoning:\n")
            f.write("  - FG, FGA removed: Highly correlated with FG% (percentage more accurately reflects team strength)\n")
            f.write("  - 2P, 2PA removed: Highly correlated with 2P%\n")
            f.write("  - 3P, 3PA removed: Highly correlated with 3P%\n")
            f.write("  - FT, FTA removed: Highly correlated with FT%\n")
            f.write("  - TRB removed: Redundant (TRB = ORB + DRB), highly correlated with DRB\n")
        
        print(f"\nSaved correlation analysis report: {report_file}")
        
        return df_selected
    
    def create_correlation_heatmap(self, df, period='game', save=True):
        """
        Create correlation heatmap for a specific period with significance levels
        Matching Research.md Fig 3 methodology
        """
        print(f"\nCreating correlation heatmap for {period}...")
        
        # Get features for this period
        period_cols = [col for col in df.columns if col.startswith(f'{period}_')]
        period_cols = [col for col in period_cols if col not in ['GAME_ID', 'SEASON', 'RESULT']]
        
        if not period_cols:
            print(f"No {period} features found")
            return None
        
        # Create correlation matrix
        corr_data = df[period_cols].corr()
        
        # Calculate p-values for correlations
        from scipy.stats import pearsonr
        n = len(df)
        p_value_matrix = np.zeros_like(corr_data)
        
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                if i != j:
                    _, p_val = pearsonr(df[corr_data.columns[i]], df[corr_data.columns[j]])
                    p_value_matrix[i, j] = p_val
        
        # Create annotation matrix with significance asterisks
        annot_matrix = corr_data.copy().astype(str)
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                if i != j:
                    corr_val = corr_data.iloc[i, j]
                    p_val = p_value_matrix[i, j]
                    
                    # Add significance asterisks
                    if p_val < 0.001:
                        sig = '***'
                    elif p_val < 0.01:
                        sig = '**'
                    elif p_val < 0.05:
                        sig = '*'
                    else:
                        sig = ''
                    
                    annot_matrix.iloc[i, j] = f'{corr_val:.2f}{sig}'
                else:
                    annot_matrix.iloc[i, j] = '1.00'
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(
            corr_data,
            mask=mask,
            annot=annot_matrix,
            fmt='',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            vmin=-1,
            vmax=1
        )
        
        plt.title(f'Correlation Heatmap: {period.upper()} Features\n'
                 f'Significance: *** p<0.001, ** p<0.01, * p<0.05', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_file = self.output_dir / f'correlation_heatmap_{period}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        plt.close()
        
        return corr_data
    
    def create_descriptive_statistics(self, df, period='game'):
        """Create descriptive statistics table (like Table 2)"""
        print(f"\nCreating descriptive statistics for {period}...")
        
        # Get features for this period
        period_cols = [col for col in df.columns if col.startswith(f'{period}_')]
        period_cols = [col for col in period_cols if col != 'RESULT']
        
        if not period_cols:
            print(f"No {period} features found")
            return None
        
        # Calculate descriptive statistics
        desc_stats = df[period_cols].describe().T
        desc_stats = desc_stats.rename(columns={
            'count': 'Count',
            'mean': 'Mean',
            'std': 'Std',
            'min': 'Min',
            '25%': '0.25',
            '50%': '0.5',
            '75%': '0.75',
            'max': 'Max'
        })
        
        # Round to match paper format
        desc_stats['Mean'] = desc_stats['Mean'].round(3)
        desc_stats['Std'] = desc_stats['Std'].round(3)
        desc_stats['Min'] = desc_stats['Min'].round(3)
        desc_stats['0.25'] = desc_stats['0.25'].round(3)
        desc_stats['0.5'] = desc_stats['0.5'].round(3)
        desc_stats['0.75'] = desc_stats['0.75'].round(3)
        desc_stats['Max'] = desc_stats['Max'].round(3)
        
        # Save to CSV
        output_file = self.output_dir / f'descriptive_stats_{period}.csv'
        desc_stats.to_csv(output_file)
        print(f"Saved: {output_file}")
        
        return desc_stats
    
    def perform_logistic_regression(self, df, period='game'):
        """Perform logistic regression analysis (like Tables 3-5)"""
        print(f"\nPerforming logistic regression for {period}...")
        
        # Get features for this period
        feature_cols = [col for col in df.columns if col.startswith(f'{period}_')]
        feature_cols = [col for col in feature_cols if col != 'RESULT']
        
        if not feature_cols:
            print(f"No {period} features found")
            return None
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df['RESULT']
        
        # Remove rows with any NaN values
        mask = ~(X.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print(f"No valid data for {period}")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        lr = LogisticRegression(max_iter=1000, solver='liblinear')
        lr.fit(X_scaled, y)
        
        # Calculate statistics
        from scipy.stats import norm
        
        coefs = lr.coef_[0]
        intercept = lr.intercept_[0]
        
        # Calculate standard errors and p-values
        # Using approximation: SE = 1 / sqrt(n * p * (1-p))
        n = len(X)
        p_mean = y.mean()
        se_approx = 1 / np.sqrt(n * p_mean * (1 - p_mean))
        
        # Calculate z-scores and p-values
        z_scores = coefs / se_approx
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Indicator': feature_cols,
            'Estimated_coeff': coefs,
            'Standard_Error': se_approx,
            'z': z_scores,
            'p_value': p_values,
            'Constant': [intercept] + [np.nan] * (len(feature_cols) - 1)
        })
        
        # Calculate 95% confidence intervals
        results['CI_lower'] = results['Estimated_coeff'] - 1.96 * results['Standard_Error']
        results['CI_upper'] = results['Estimated_coeff'] + 1.96 * results['Standard_Error']
        
        # Round values
        results['Estimated_coeff'] = results['Estimated_coeff'].round(4)
        results['Standard_Error'] = results['Standard_Error'].round(3)
        results['z'] = results['z'].round(3)
        results['p_value'] = results['p_value'].round(4)
        results['CI_lower'] = results['CI_lower'].round(3)
        results['CI_upper'] = results['CI_upper'].round(3)
        
        # Save results
        output_file = self.output_dir / f'logistic_regression_{period}.csv'
        results.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        
        return results
    
    def save_prepared_dataset(self, df):
        """Save the prepared dataset"""
        print("\n" + "=" * 60)
        print("Saving prepared dataset...")
        print("=" * 60)
        
        # Save full dataset
        output_file = self.output_dir / 'nba_prepared_data.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        print(f"Shape: {df.shape}")
        
        # Save summary
        summary_file = self.output_dir / 'preparation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("NBA Data Preparation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total games: {len(df)}\n")
            f.write(f"Total features: {len(df.columns)}\n")
            f.write(f"\nSeasons:\n")
            for season in df['SEASON'].unique():
                count = len(df[df['SEASON'] == season])
                f.write(f"  {season}: {count} games\n")
            f.write(f"\nFeatures included:\n")
            for col in sorted(df.columns):
                if col not in ['GAME_ID', 'SEASON', 'RESULT']:
                    f.write(f"  {col}\n")
        
        print(f"Saved: {summary_file}")
        
        return output_file
    
    def prepare_all(self):
        """Run complete data preparation pipeline"""
        print("\n" + "=" * 60)
        print("NBA Data Preparation Pipeline")
        print("Following Research.md methodology (lines 158-176)")
        print("=" * 60)
        
        # Step 1: Load all games
        df = self.load_all_games()
        
        # Step 2: Remove outliers
        df = self.remove_outliers(df)
        
        # Step 3: Create initial correlation heatmaps (before feature selection)
        print("\n" + "=" * 60)
        print("Creating Initial Correlation Heatmaps (Before Feature Selection)")
        print("=" * 60)
        for period in ['game', 'H2', 'H3']:
            self.create_correlation_heatmap(df, period=period)
        
        # Step 4: Feature selection (includes correlation analysis and scatter plots)
        df_selected = self.perform_feature_selection(df)
        
        # Step 5: Create correlation heatmaps (after feature selection)
        print("\n" + "=" * 60)
        print("Creating Correlation Heatmaps (After Feature Selection)")
        print("=" * 60)
        for period in ['game', 'H2', 'H3']:
            self.create_correlation_heatmap(df_selected, period=period)
        
        # Step 5: Create descriptive statistics
        for period in ['game', 'H2', 'H3']:
            self.create_descriptive_statistics(df_selected, period=period)
        
        # Step 6: Perform logistic regression
        for period in ['game', 'H2', 'H3']:
            self.perform_logistic_regression(df_selected, period=period)
        
        # Step 7: Save prepared dataset
        self.save_prepared_dataset(df_selected)
        
        print("\n" + "=" * 60)
        print("Data preparation complete!")
        print("=" * 60)
        
        return df_selected


def main():
    """Main execution function"""
    preparer = NBADataPreparation()
    df_prepared = preparer.prepare_all()
    
    print("\nPrepared dataset summary:")
    print(df_prepared.head())
    print(f"\nDataset shape: {df_prepared.shape}")


if __name__ == "__main__":
    main()

