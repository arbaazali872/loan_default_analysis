from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import yaml
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.features import create_derived_features, apply_log_transform, drop_correlated_features

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting car loan defaults",
    version="1.0.0"
)

# Global variables for model and preprocessor
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
    
    logger.info("Model and preprocessor loaded successfully")

# Load artifacts on startup
@app.on_event("startup")
async def startup_event():
    load_artifacts()
    logger.info("API started successfully")

# Request schema
class LoanApplication(BaseModel):
    disbursed_amount: float = Field(..., description="Amount disbursed")
    asset_cost: float = Field(..., description="Cost of asset")
    ltv: float = Field(..., description="Loan to value ratio")
    branch_id: int = Field(..., description="Branch ID")
    supplier_id: int = Field(..., description="Supplier ID")
    manufacturer_id: int = Field(..., description="Manufacturer ID")
    current_pincode_id: int = Field(..., description="Current pincode ID")
    date_of_birth: str = Field(..., description="Date of birth (DD/MM/YYYY or DD-MM-YY)")
    employment_type: str = Field(..., description="Employment type")
    state_id: int = Field(..., description="State ID")
    employee_code_id: int = Field(..., description="Employee code ID")
    mobileno_avl_flag: int = Field(..., description="Mobile number available flag")
    aadhar_flag: int = Field(..., description="Aadhar flag")
    pan_flag: int = Field(..., description="PAN flag")
    voterid_flag: int = Field(..., description="Voter ID flag")
    driving_flag: int = Field(..., description="Driving license flag")
    passport_flag: int = Field(..., description="Passport flag")
    pri_no_of_accts: int = Field(..., description="Primary number of accounts")
    pri_active_accts: int = Field(..., description="Primary active accounts")
    pri_overdue_accts: int = Field(..., description="Primary overdue accounts")
    pri_current_balance: float = Field(..., description="Primary current balance")
    pri_sanctioned_amount: float = Field(..., description="Primary sanctioned amount")
    pri_disbursed_amount: float = Field(..., description="Primary disbursed amount")
    sec_no_of_accts: int = Field(..., description="Secondary number of accounts")
    sec_active_accts: int = Field(..., description="Secondary active accounts")
    sec_overdue_accts: int = Field(..., description="Secondary overdue accounts")
    sec_current_balance: float = Field(..., description="Secondary current balance")
    sec_sanctioned_amount: float = Field(..., description="Secondary sanctioned amount")
    sec_disbursed_amount: float = Field(..., description="Secondary disbursed amount")
    primary_instal_amt: float = Field(..., description="Primary installment amount")
    sec_instal_amt: float = Field(..., description="Secondary installment amount")
    new_accts_in_last_six_months: int = Field(..., description="New accounts in last 6 months")
    delinquent_accts_in_last_six_months: int = Field(..., description="Delinquent accounts in last 6 months")
    average_acct_age: str = Field(..., description="Average account age (e.g., '3yrs 6mon')")
    credit_history_length: str = Field(..., description="Credit history length (e.g., '5yrs 2mon')")
    no_of_inquiries: int = Field(..., description="Number of inquiries")
    perform_cns_score: float = Field(..., description="Performance score")
    perform_cns_score_description: str = Field(..., description="Performance score description")

    class Config:
        schema_extra = {
            "example": {
                "disbursed_amount": 95000,
                "asset_cost": 100000,
                "ltv": 95.0,
                "branch_id": 123,
                "supplier_id": 456,
                "manufacturer_id": 789,
                "current_pincode_id": 110001,
                "date_of_birth": "15/06/1990",
                "employment_type": "Salaried",
                "state_id": 10,
                "employee_code_id": 555,
                "mobileno_avl_flag": 1,
                "aadhar_flag": 1,
                "pan_flag": 1,
                "voterid_flag": 0,
                "driving_flag": 1,
                "passport_flag": 0,
                "pri_no_of_accts": 5,
                "pri_active_accts": 3,
                "pri_overdue_accts": 0,
                "pri_current_balance": 50000,
                "pri_sanctioned_amount": 200000,
                "pri_disbursed_amount": 180000,
                "sec_no_of_accts": 2,
                "sec_active_accts": 1,
                "sec_overdue_accts": 0,
                "sec_current_balance": 20000,
                "sec_sanctioned_amount": 50000,
                "sec_disbursed_amount": 45000,
                "primary_instal_amt": 5000,
                "sec_instal_amt": 1500,
                "new_accts_in_last_six_months": 1,
                "delinquent_accts_in_last_six_months": 0,
                "average_acct_age": "3yrs 6mon",
                "credit_history_length": "5yrs 2mon",
                "no_of_inquiries": 2,
                "perform_cns_score": 750.0,
                "perform_cns_score_description": "Good"
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0: No default, 1: Default")
    probability: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Low, Medium, or High risk")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Loan Default Prediction API is running"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """Predict loan default probability"""
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([application.dict()])
        
        # Clean column names
        input_data.columns = (
            input_data.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        
        # Apply feature engineering
        input_data = create_derived_features(input_data)
        input_data = apply_log_transform(input_data, config['feature_engineering']['log_transform_features'])
        input_data = drop_correlated_features(input_data)
        
        # Remove ID columns and target if present
        cols_to_drop = ['uniqueid', 'loan_default'] + [col for col in input_data.columns if '_id' in col]
        input_data = input_data.drop(columns=cols_to_drop, errors='ignore')
        
        # Preprocess
        X_processed = preprocessor.transform(input_data)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        logger.info(f"Prediction made: {prediction}, Probability: {probability:.4f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)