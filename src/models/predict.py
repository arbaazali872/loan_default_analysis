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
    """Load trained model pipeline"""
    model_dir = Path(config['artifacts']['model_path'])
    
   
    model = joblib.load(model_dir / config['artifacts']['model_filename'])
    
    logger.info("Model loaded successfully")
    
    return model

def predict(data: pd.DataFrame, model):
    """Make predictions on new data"""
    logger.info(f"Making predictions on {len(data)} records")
    

    predictions = model.predict(data)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data)[:, 1]
        return predictions, probabilities
    
    return predictions, None

def predict_single(features: dict, model):
    """Make prediction on single instance"""
    df = pd.DataFrame([features])
    predictions, probabilities = predict(df, model)
    
    return predictions[0], probabilities[0] if probabilities is not None else None

if __name__ == "__main__":
    # Example usage
    config = load_config()
    model = load_artifacts(config)
    
    # Load test data or create sample
    # predictions, probabilities = predict(test_data, model)
    
    logger.info("Prediction module ready")