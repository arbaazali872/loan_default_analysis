import yaml
import joblib
import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_artifacts(config):
    """Load trained model and preprocessor"""
    model_dir = Path(config['artifacts']['model_path'])
    
    preprocessor = joblib.load(model_dir / config['artifacts']['preprocessor_filename'])
    model = joblib.load(model_dir / config['artifacts']['model_filename'])
    
    logger.info("Model and preprocessor loaded successfully")
    
    return preprocessor, model

def predict(data: pd.DataFrame, preprocessor, model):
    """Make predictions on new data"""
    logger.info(f"Making predictions on {len(data)} records")
    
    # Preprocess data
    X_processed = preprocessor.transform(data)
    
    # Predict
    predictions = model.predict(X_processed)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_processed)[:, 1]
        return predictions, probabilities
    
    return predictions, None

def predict_single(features: dict, preprocessor, model):
    """Make prediction on single instance"""
    df = pd.DataFrame([features])
    predictions, probabilities = predict(df, preprocessor, model)
    
    return predictions[0], probabilities[0] if probabilities is not None else None

if __name__ == "__main__":
    # Example usage
    config = load_config()
    preprocessor, model = load_artifacts(config)
    
    # Load test data or create sample
    # predictions, probabilities = predict(test_data, preprocessor, model)
    
    logger.info("Prediction module ready")