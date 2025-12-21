import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def convert_time_to_years(series: pd.Series, column_name: str) -> pd.Series:
    """Convert time format (Xyrs Ymon) to years"""
    logger.info(f"Converting {column_name} to years")
    
    years = series.str.extract(r'(\d+)').astype(float)
    months = series.str.extract(r'(\d+)m').fillna(0).astype(float) / 12
    
    return years + months

def convert_dates(date_series: pd.Series) -> pd.Series:
    """Convert date strings to datetime"""
    logger.info("Converting dates")
    
    converted_dates_1 = pd.to_datetime(date_series, format='%d/%m/%Y', errors='coerce')
    converted_dates_2 = pd.to_datetime(date_series, format='%d-%m-%y', errors='coerce')
    
    return converted_dates_1.fillna(converted_dates_2)

def calculate_age(birth_year: pd.Series, reference_year: int = 2024) -> pd.Series:
    """Calculate age from birth year"""
    logger.info("Calculating customer age")
    
    age = reference_year - birth_year
    # Fix negative ages
    age = np.where(age < 0, age.median(), age)
    
    return age

def apply_log_transform(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Apply log transformation to reduce skewness"""
    logger.info(f"Applying log transformation to {len(features)} features")
    
    for feature in features:
        if feature in df.columns:
            # Replace negative values with 0 before log transform
            df[feature] = df[feature].clip(lower=0)
            # Apply log1p which handles 0 values safely
            df[feature] = np.log1p(df[feature])
            # Replace any remaining inf/-inf with NaN (will be imputed later)
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
    
    return df

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features"""
    logger.info("Creating derived features")
    
    # Time-based features
    df['avg_acct_age'] = convert_time_to_years(df['average_acct_age'], 'average_acct_age')
    df['credit_hist_leng'] = convert_time_to_years(df['credit_history_length'], 'credit_history_length')
    
    # Date-based features
    df['date_of_birth'] = convert_dates(df['date_of_birth'])
    df['birth_year'] = df['date_of_birth'].dt.year
    df['customer_age'] = calculate_age(df['birth_year'])
    
    # ID verification score
    df['id_verification_score'] = (
        df['mobileno_avl_flag'] + 
        df['aadhar_flag'] + 
        df['pan_flag'] + 
        df['voterid_flag'] + 
        df['driving_flag'] + 
        df['passport_flag']
    )
    
    # Financial ratios
    df['loan_burden_ratio'] = (
        (df['primary_instal_amt'] + df['sec_instal_amt']) / df['asset_cost']
    )
    
    # Credit behavior
    df['new_credit_behavior'] = (
        df['new_accts_in_last_six_months'] + df['no_of_inquiries']
    )
    
    # Credit stability
    df['credit_stability'] = df['credit_hist_leng'] + df['avg_acct_age']
    
    # Drop original columns that were transformed
    df = df.drop(columns=[
        'average_acct_age', 
        'credit_history_length', 
        'date_of_birth'
    ])
    
    # Fill missing employment type
    df['employment_type'].fillna('Unknown', inplace=True)
    
    logger.info("Derived features created successfully")
    
    return df

def drop_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop highly correlated features based on domain analysis"""
    logger.info("Dropping highly correlated features")
    
    features_to_drop = [
        'sec_instal_amt',
        'pri_current_balance',
        'sec_no_of_accts',
        'pri_active_accts',
        'sec_sanctioned_amount'
    ]
    
    existing_features = [f for f in features_to_drop if f in df.columns]
    df = df.drop(columns=existing_features)
    
    logger.info(f"Dropped {len(existing_features)} correlated features")
    
    return df