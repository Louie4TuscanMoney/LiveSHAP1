"""
NBA Model Training Script
Follows the methodology from Research.md (lines 177-212)

This script:
1. Loads prepared data and groups by period (H2, H3, game)
2. Trains multiple ML algorithms with hyperparameter tuning
3. Performs 10-fold cross-validation
4. Evaluates using AUC, F1 Score, accuracy, precision, recall
5. Uses SHAP for interpretability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

warnings.filterwarnings('ignore')


class NBAModelTraining:
    """
    Train and evaluate NBA game outcome prediction models
    Following Research.md methodology (lines 177-212)
    """
    
    def __init__(self, prepared_data_path='2datapreparation/nba_prepared_data.csv', 
                 output_dir='3modeltraining'):
        self.prepared_data_path = Path(prepared_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # Load prepared data
        self.df = pd.read_csv(self.prepared_data_path)
        
        # Initialize models
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def prepare_datasets_by_period(self):
        """
        Group datasets by period: H2, H3, and full game
        Returns three separate datasets for each period
        """
        print("=" * 60)
        print("Preparing Datasets by Period")
        print("=" * 60)
        
        datasets = {}
        
        for period in ['H2', 'H3', 'game']:
            # Get features for this period
            period_cols = [col for col in self.df.columns if col.startswith(f'{period}_')]
            period_cols = [col for col in period_cols if col not in ['GAME_ID', 'SEASON', 'RESULT']]
            
            # Create dataset for this period
            X = self.df[period_cols].fillna(0)
            y = self.df['RESULT']
            
            # Remove rows with any NaN values
            mask = ~(X.isna().any(axis=1))
            X = X[mask]
            y = y[mask]
            
            datasets[period] = {
                'X': X,
                'y': y,
                'feature_names': period_cols
            }
            
            print(f"\n{period.upper()} Dataset:")
            print(f"  Samples: {len(X)}")
            print(f"  Features: {len(period_cols)}")
            print(f"  Positive class (home win): {y.sum()} ({y.mean()*100:.1f}%)")
        
        return datasets
    
    def get_model_configs(self):
        """Get model configurations for hyperparameter tuning"""
        return {
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'bayesian_params': {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(0, 10),
                    'reg_lambda': Real(0, 10)
                },
                'grid_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'bayesian_params': {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.6, 1.0),
                    'colsample_bytree': Real(0.6, 1.0),
                    'reg_alpha': Real(0, 10),
                    'reg_lambda': Real(0, 10)
                },
                'grid_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'bayesian_params': {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                },
                'grid_params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'bayesian_params': {
                    'C': Real(0.1, 100, prior='log-uniform'),
                    'gamma': Real(0.001, 1, prior='log-uniform'),
                    'kernel': ['rbf', 'poly', 'sigmoid']
                },
                'grid_params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'bayesian_params': {
                    'n_neighbors': Integer(3, 20),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'grid_params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'bayesian_params': {
                    'C': Real(0.01, 100, prior='log-uniform'),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'grid_params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'bayesian_params': {
                    'max_depth': Integer(3, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'criterion': ['gini', 'entropy']
                },
                'grid_params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            }
        }
    
    def perform_hyperparameter_tuning(self, model_name, model_config, X, y, cv=5):
        """
        Perform hyperparameter tuning using Bayesian optimization and grid search
        Returns best model
        """
        print(f"\n  Tuning {model_name}...")
        
        # Reduce iterations for slow models (SVM)
        if model_name == 'SVM':
            n_iter = 10  # Reduced from 30 for SVM (much faster)
            cv_folds = 3  # Reduced from 5 for SVM
            print(f"    Using reduced tuning (n_iter={n_iter}, cv={cv_folds}) for faster SVM training...")
        else:
            n_iter = 30
            cv_folds = cv
        
        # Try Bayesian optimization first (faster, more efficient)
        try:
            print(f"    Using Bayesian optimization...")
            bayesian_search = BayesSearchCV(
                model_config['model'],
                model_config['bayesian_params'],
                n_iter=n_iter,  # Number of iterations (reduced for SVM)
                cv=cv_folds,  # CV folds (reduced for SVM)
                scoring='roc_auc',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            bayesian_search.fit(X, y)
            best_model = bayesian_search.best_estimator_
            best_score = bayesian_search.best_score_
            print(f"    Best CV score (Bayesian): {best_score:.4f}")
        except Exception as e:
            print(f"    Bayesian optimization failed: {e}")
            print(f"    Falling back to grid search...")
            # Fall back to grid search
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['grid_params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            print(f"    Best CV score (Grid): {best_score:.4f}")
        
        return best_model
    
    def evaluate_model(self, model, X, y, model_name, cv=10):
        """
        Evaluate model using 10-fold cross-validation
        Returns metrics: AUC, F1 Score, accuracy, precision, recall
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Determine if model needs scaling
        needs_scaling = model_name in ['SVM', 'Logistic Regression', 'KNN']
        
        metrics = {
            'AUC': [],
            'F1_Score': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': []
        }
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features for algorithms that need it
            if needs_scaling:
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_val = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
            
            # Create a fresh model instance for each fold
            model_fold = type(model)(**model.get_params())
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_val)
            y_pred_proba = model_fold.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics['AUC'].append(roc_auc_score(y_val, y_pred_proba))
            metrics['F1_Score'].append(f1_score(y_val, y_pred))
            metrics['Accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['Precision'].append(precision_score(y_val, y_pred))
            metrics['Recall'].append(recall_score(y_val, y_pred))
        
        # Calculate mean and std for each metric
        results = {}
        for metric_name, values in metrics.items():
            results[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return results
    
    def train_and_evaluate_all_models(self, datasets, use_hyperparameter_tuning=True):
        """
        Train and evaluate all models for all periods
        """
        print("\n" + "=" * 60)
        print("Training and Evaluating Models")
        print("=" * 60)
        
        model_configs = self.get_model_configs()
        all_results = {}
        
        for period in ['H2', 'H3', 'game']:
            print(f"\n{'='*60}")
            print(f"Period: {period.upper()}")
            print(f"{'='*60}")
            
            X = datasets[period]['X']
            y = datasets[period]['y']
            feature_names = datasets[period]['feature_names']
            
            period_results = {}
            
            for model_name, model_config in model_configs.items():
                print(f"\n{model_name}:")
                
                # Skip SVM by default (slow, not a top performer per research paper)
                # Set to False to include SVM (matches research paper exactly)
                SKIP_SVM = True  # Default: True (skip SVM for faster training)
                if SKIP_SVM and model_name == 'SVM':
                    print(f"  Skipping SVM (slow, not a top performer)")
                    continue
                
                # Hyperparameter tuning
                if use_hyperparameter_tuning:
                    best_model = self.perform_hyperparameter_tuning(
                        model_name, model_config, X, y, cv=5
                    )
                else:
                    best_model = model_config['model']
                
                # 10-fold cross-validation evaluation
                print(f"  Performing 10-fold cross-validation...")
                cv_results = self.evaluate_model(best_model, X, y, model_name, cv=10)
                
                period_results[model_name] = {
                    'model': best_model,
                    'metrics': cv_results
                }
                
                # Print results
                print(f"  Results:")
                for metric_name, values in cv_results.items():
                    print(f"    {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")
            
            all_results[period] = period_results
            
            # Save best model for this period (XGBoost)
            if 'XGBoost' in period_results:
                best_xgb = period_results['XGBoost']['model']
                model_file = self.models_dir / f'xgb_model_{period}.pkl'
                joblib.dump(best_xgb, model_file)
                print(f"\n  Saved best XGBoost model: {model_file}")
        
        self.results = all_results
        return all_results
    
    def create_results_tables(self, results):
        """
        Create results tables matching Tables 7-9 from the paper
        """
        print("\n" + "=" * 60)
        print("Creating Results Tables")
        print("=" * 60)
        
        for period in ['H2', 'H3', 'game']:
            period_name = {
                'H2': 'First Two Quarters',
                'H3': 'First Three Quarters',
                'game': 'Full Game'
            }[period]
            
            print(f"\n{period_name} Period Results:")
            
            # Create results DataFrame
            rows = []
            for model_name, model_data in results[period].items():
                metrics = model_data['metrics']
                rows.append({
                    'Algorithm': model_name,
                    'AUC': f"{metrics['AUC']['mean']:.3f}",
                    'F1_Score': f"{metrics['F1_Score']['mean']:.3f}",
                    'Accuracy': f"{metrics['Accuracy']['mean']:.3f}",
                    'Precision': f"{metrics['Precision']['mean']:.3f}",
                    'Recall': f"{metrics['Recall']['mean']:.3f}"
                })
            
            df_results = pd.DataFrame(rows)
            df_results = df_results.sort_values('AUC', ascending=False)
            
            # Save to CSV
            output_file = self.output_dir / f'results_{period}.csv'
            df_results.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
            
            # Print table
            print(df_results.to_string(index=False))
    
    def create_comparison_chart(self, results):
        """
        Create comparison chart matching Fig 5 from the paper
        """
        print("\nCreating comparison charts...")
        
        metrics = ['AUC', 'F1_Score', 'Accuracy', 'Precision', 'Recall']
        periods = ['H2', 'H3', 'game']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, period in enumerate(periods):
            period_name = {
                'H2': 'First Two Quarters',
                'H3': 'First Three Quarters',
                'game': 'Full Game'
            }[period]
            
            # Prepare data
            model_names = []
            metric_values = {metric: [] for metric in metrics}
            
            for model_name, model_data in results[period].items():
                model_names.append(model_name)
                for metric in metrics:
                    metric_values[metric].append(model_data['metrics'][metric]['mean'])
            
            # Create grouped bar chart
            x = np.arange(len(model_names))
            width = 0.15
            
            for i, metric in enumerate(metrics):
                axes[idx].bar(x + i*width, metric_values[metric], width, label=metric)
            
            axes[idx].set_xlabel('Algorithm', fontsize=10)
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].set_title(f'{period_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x + width * 2)
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_ylim([0, 1])
        
        plt.suptitle('Model Performance Comparison Across Periods', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'model_comparison_chart.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def perform_shap_analysis(self, datasets, results):
        """
        Perform SHAP analysis on best XGBoost models
        """
        print("\n" + "=" * 60)
        print("SHAP Interpretability Analysis")
        print("=" * 60)
        
        for period in ['H2', 'H3', 'game']:
            print(f"\n{period.upper()} Period:")
            
            if 'XGBoost' not in results[period]:
                print(f"  XGBoost model not found, skipping SHAP analysis")
                continue
            
            X = datasets[period]['X']
            y = datasets[period]['y']
            feature_names = datasets[period]['feature_names']
            model = results[period]['XGBoost']['model']
            
            # Prepare sample for SHAP (needed for fallback KernelExplainer)
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            
            # Create SHAP explainer
            print(f"  Creating SHAP explainer...")
            
            # Fix XGBoost base_score issue before creating explainer
            if hasattr(model, 'get_booster'):
                try:
                    booster = model.get_booster()
                    config_str = booster.save_config()
                    config_dict = json.loads(config_str)
                    if 'learner' in config_dict and 'learner_model_param' in config_dict['learner']:
                        base_score = config_dict['learner']['learner_model_param'].get('base_score', '0.5')
                        if isinstance(base_score, str) and base_score.startswith('['):
                            # Extract float from string array like '[5.111304E-1]'
                            base_score_val = float(base_score.strip('[]'))
                            config_dict['learner']['learner_model_param']['base_score'] = str(base_score_val)
                            booster.load_config(json.dumps(config_dict))
                            print(f"  Fixed base_score: {base_score} → {base_score_val}")
                except Exception as e:
                    print(f"  Warning: Could not fix base_score: {e}")
            
            try:
                explainer = shap.TreeExplainer(model)
            except Exception as e:
                # Fallback: use model_output='probability' or try different approach
                print(f"  Warning: SHAP TreeExplainer failed: {e}")
                print(f"  Trying alternative SHAP explainer...")
                try:
                    # Try with model_output explicitly set
                    explainer = shap.TreeExplainer(model, model_output='probability')
                except:
                    # Last resort: use KernelExplainer (slower but more robust)
                    print(f"  Using KernelExplainer (slower but more robust)...")
                    explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            
            # Calculate SHAP values (handle different explainer types)
            if isinstance(explainer, shap.explainers.Kernel):
                shap_values = explainer.shap_values(X_sample)[:, 1]  # Get class 1 probabilities
            else:
                shap_values = explainer.shap_values(X_sample)
                # Handle binary classification - get positive class SHAP values
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Use positive class (home win)
            
            # SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot: {period.upper()} Period', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            output_file = self.output_dir / f'shap_summary_{period}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_file}")
            plt.close()
            
            # Calculate feature importance (mean absolute SHAP values)
            shap_importance = pd.DataFrame({
                'Rank': range(1, len(feature_names) + 1),
                'Indicators': feature_names,
                'Values': np.abs(shap_values).mean(axis=0)
            }).sort_values('Values', ascending=False)
            shap_importance['Rank'] = range(1, len(shap_importance) + 1)
            
            # Save feature importance (matching Table 10 format)
            importance_file = self.output_dir / f'shap_importance_{period}.csv'
            shap_importance[['Rank', 'Indicators', 'Values']].to_csv(importance_file, index=False)
            print(f"  Saved: {importance_file}")
            
            print(f"  Top 5 features (SHAP importance):")
            for idx, row in shap_importance.head(5).iterrows():
                print(f"    {row['Rank']}. {row['Indicators']}: {row['Values']:.4f}")
            
            # Store for comparison table
            results[period]['XGBoost']['shap_importance'] = shap_importance
    
    def train_all(self, use_hyperparameter_tuning=True):
        """
        Run complete model training pipeline
        """
        print("\n" + "=" * 60)
        print("NBA Model Training Pipeline")
        print("Following Research.md methodology (lines 177-212)")
        print("=" * 60)
        
        # Step 1: Prepare datasets by period
        datasets = self.prepare_datasets_by_period()
        
        # Step 2: Train and evaluate all models
        results = self.train_and_evaluate_all_models(datasets, use_hyperparameter_tuning)
        
        # Step 3: Create results tables
        self.create_results_tables(results)
        
        # Step 4: Create comparison charts
        self.create_comparison_chart(results)
        
        # Step 5: SHAP analysis
        self.perform_shap_analysis(datasets, results)
        
        # Step 6: Create SHAP comparison table (Table 10)
        self.create_shap_comparison_table(results)
        
        # Step 7: Create SHAP summary chart (Fig 6)
        self.create_shap_summary_chart(results)
        
        print("\n" + "=" * 60)
        print("Model training complete!")
        print("=" * 60)
        
        return results
    
    def create_shap_summary_chart(self, results):
        """
        Create SHAP feature importance summary chart (matching Fig 6)
        Shows feature importance rankings across different periods
        """
        print("\nCreating SHAP summary chart (Fig 6)...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        for idx, period in enumerate(['H2', 'H3', 'game']):
            period_name = {
                'H2': 'First Two Quarters',
                'H3': 'First Three Quarters',
                'game': 'Full Game'
            }[period]
            
            if 'XGBoost' in results[period] and 'shap_importance' in results[period]['XGBoost']:
                shap_df = results[period]['XGBoost']['shap_importance'].head(11)
                
                # Create horizontal bar chart
                axes[idx].barh(range(len(shap_df)), shap_df['Values'], color='steelblue')
                axes[idx].set_yticks(range(len(shap_df)))
                axes[idx].set_yticklabels(shap_df['Indicators'], fontsize=9)
                axes[idx].set_xlabel('SHAP Importance Value', fontsize=10)
                axes[idx].set_title(f'{period_name}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='x')
                axes[idx].invert_yaxis()  # Top feature at top
            else:
                axes[idx].text(0.5, 0.5, 'No SHAP data', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{period_name}', fontsize=12, fontweight='bold')
        
        plt.suptitle('SHAP Feature Importance Summary Across Periods', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / 'shap_summary_chart.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def create_shap_comparison_table(self, results):
        """
        Create SHAP feature importance comparison table (matching Table 10)
        Shows feature rankings across different periods
        """
        print("\n" + "=" * 60)
        print("Creating SHAP Feature Importance Comparison Table")
        print("=" * 60)
        
        # Get all unique features across all periods
        all_features = set()
        for period in ['H2', 'H3', 'game']:
            if 'XGBoost' in results[period] and 'shap_importance' in results[period]['XGBoost']:
                features = results[period]['XGBoost']['shap_importance']['Indicators'].tolist()
                all_features.update(features)
        
        # Create comparison DataFrame
        comparison_data = []
        for rank in range(1, 12):  # Top 11 features (matching paper)
            row = {'Rank': rank}
            for period in ['H2', 'H3', 'game']:
                period_name = {
                    'H2': 'First Two Quarters',
                    'H3': 'First Three Quarters',
                    'game': 'Full Game'
                }[period]
                
                if 'XGBoost' in results[period] and 'shap_importance' in results[period]['XGBoost']:
                    shap_df = results[period]['XGBoost']['shap_importance']
                    if rank <= len(shap_df):
                        feature_row = shap_df.iloc[rank - 1]
                        row[f'{period_name}_Indicator'] = feature_row['Indicators']
                        row[f'{period_name}_Value'] = f"{feature_row['Values']:.4f}"
                    else:
                        row[f'{period_name}_Indicator'] = '-'
                        row[f'{period_name}_Value'] = '-'
                else:
                    row[f'{period_name}_Indicator'] = '-'
                    row[f'{period_name}_Value'] = '-'
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        output_file = self.output_dir / 'shap_comparison_table.csv'
        comparison_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        
        # Print table
        print("\nSHAP Feature Importance Comparison (Top 11):")
        print(comparison_df.to_string(index=False))


def main():
    """Main execution function"""
    trainer = NBAModelTraining()
    results = trainer.train_all(use_hyperparameter_tuning=True)
    
    print("\nTraining Summary:")
    for period in ['H2', 'H3', 'game']:
        print(f"\n{period.upper()} Period - Best Model (XGBoost):")
        xgb_results = results[period]['XGBoost']['metrics']
        for metric_name, values in xgb_results.items():
            print(f"  {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")


if __name__ == "__main__":
    main()

