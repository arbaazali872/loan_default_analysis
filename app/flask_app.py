from flask import Flask, request, jsonify
import joblib
import yaml
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.features import create_derived_features, apply_log_transform, drop_correlated_features
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
config = None

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_artifacts():
    """Load model and preprocessor"""
    global model, preprocessor, config
    
    config = load_config()
    model_dir = Path(__file__).parent.parent / config['artifacts']['model_path']
    
    preprocessor = joblib.load(model_dir / config['artifacts']['preprocessor_filename'])
    model = joblib.load(model_dir / config['artifacts']['model_filename'])
    
    logger.info("Model and preprocessor loaded")

# Load on startup
load_artifacts()

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict loan default"""
    try:
        data = request.json
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Clean columns
        input_data.columns = input_data.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Feature engineering
        input_data = create_derived_features(input_data)
        input_data = apply_log_transform(input_data, config['feature_engineering']['log_transform_features'])
        input_data = drop_correlated_features(input_data)
        
        # Remove ID columns
        cols_to_drop = ['uniqueid', 'loan_default'] + [col for col in input_data.columns if '_id' in col]
        input_data = input_data.drop(columns=cols_to_drop, errors='ignore')
        
        # Model is full pipeline, predict directly
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_no_default': float(probability[0]),
            'probability_default': float(probability[1])
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)