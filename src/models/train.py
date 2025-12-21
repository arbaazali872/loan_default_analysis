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

def train_model(model, model_name, X_train, y_train, X_test, y_test, config):
    """Train and log model with MLflow"""
    logger.info(f"Training {model_name}")
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(model.get_params())
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
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
        
        # ROC AUC if predict_proba available
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            train_metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
            test_metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
        
        # Log metrics
        mlflow.log_metrics({**train_metrics, **test_metrics})
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"{model_name} - Test Accuracy: {test_metrics['test_accuracy']:.4f}, Test F1: {test_metrics['test_f1']:.4f}")
        
        return model, test_metrics

def main():
    """Main training pipeline"""
    # Load config
    config = load_config()
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(config)
    
    # Get feature types
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features, config)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=config['smote']['sampling_strategy'],
        random_state=config['smote']['random_state']
    )
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    
    logger.info(f"After SMOTE - Train shape: {X_train_smote.shape}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(**config['models']['logistic_regression']),
        'Random Forest': RandomForestClassifier(**config['models']['random_forest']),
        'Decision Tree': DecisionTreeClassifier(**config['models']['decision_tree'])
    }
    
    # Train all models
    results = {}
    for model_name, model in models.items():
        trained_model, metrics = train_model(
            model, model_name,
            X_train_smote, y_train_smote,
            X_test_processed, y_test,
            config
        )
        results[model_name] = {'model': trained_model, 'metrics': metrics}
    
    # Save best model and preprocessor
    best_model_name = max(results, key=lambda x: results[x]['metrics']['test_f1'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name}")
    
    # Save artifacts
    model_dir = Path(config['artifacts']['model_path'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, model_dir / config['artifacts']['preprocessor_filename'])
    joblib.dump(best_model, model_dir / config['artifacts']['model_filename'])
    
    logger.info(f"Model and preprocessor saved to {model_dir}")

if __name__ == "__main__":
    main()