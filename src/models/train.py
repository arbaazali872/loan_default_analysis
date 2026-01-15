import yaml
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.data import load_raw_data, clean_column_names, remove_id_columns, create_preprocessor
from src.features import create_derived_features, apply_log_transform, drop_correlated_features
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config):
    """Load and prepare data"""
    logger.info("Starting data preparation")
    
    # Load data
    df = load_raw_data(config['data']['raw_path'])
    df = clean_column_names(df)
    df = remove_id_columns(df)
    
    # Feature engineering
    df = create_derived_features(df)
    df = apply_log_transform(df, config['feature_engineering']['log_transform_features'])
    df = drop_correlated_features(df)
    
    # Separate features and target
    X = df.drop(columns=['uniqueid', 'loan_default'], errors='ignore')
    y = df['loan_default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    logger.info(f"Data prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model_with_tuning(model_name, model, param_grid, X_train, y_train, X_test, y_test, config):
    """Train model using GridSearchCV with SMOTE inside a pipeline"""
    logger.info(f"Training {model_name} with hyperparameter tuning")
    
    # Build pipeline: Preprocessor + SMOTE + Model
    pipeline = Pipeline([
        ('preprocessor', create_preprocessor(
            X_train.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            X_train.select_dtypes(include=['object']).columns.tolist(),
            config
        )),
        ('smote', SMOTE(
            sampling_strategy=config['smote']['sampling_strategy'],
            random_state=config['smote']['random_state']
        )),
        ('model', model)
    ])
    
    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='recall',  # prioritize detecting defaults
        verbose=1,
        n_jobs=2
    )
    
    grid_search.fit(X_train, y_train)  # raw data, pipeline handles SMOTE
    
    best_model = grid_search.best_estimator_
    
    # Predictions and metrics
    y_test_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)
    
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred)
    }
    
    if hasattr(best_model, 'predict_proba'):
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        train_metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
        test_metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
    
    # Log with MLflow
    with mlflow.start_run(run_name=f"{model_name}_Tuned"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_best_score", grid_search.best_score_)
        mlflow.log_metrics({**train_metrics, **test_metrics})
        mlflow.sklearn.log_model(best_model, "model")
    
    logger.info(f"{model_name} - CV Best Recall: {grid_search.best_score_:.4f}")
    logger.info(f"{model_name} - Test Metrics: {test_metrics}")
    logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
    
    return best_model, test_metrics

def main():
    """Main training pipeline"""
    # Load config
    config = load_config()
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(config)
    
    # Define models with parameter grids
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'param_grid': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga'],
                'model__class_weight': [None, 'balanced']
            }
        },
        # Add other models here if needed
    }
    
    results = {}
    for model_name, model_cfg in models_config.items():
        trained_model, metrics = train_model_with_tuning(
            model_name,
            model_cfg['model'],
            model_cfg['param_grid'],
            X_train, y_train,
            X_test, y_test,
            config
        )
        results[model_name] = {'model': trained_model, 'metrics': metrics}
    
    # Save best model and preprocessor
    best_model_name = max(results, key=lambda x: results[x]['metrics']['test_recall'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name}")
    
    # Save artifacts
    model_dir = Path(config['artifacts']['model_path'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model.named_steps['preprocessor'], model_dir / config['artifacts']['preprocessor_filename'])
    joblib.dump(best_model, model_dir / config['artifacts']['model_filename'])
    
    logger.info(f"Model and preprocessor saved to {model_dir}")

if __name__ == "__main__":
    main()
