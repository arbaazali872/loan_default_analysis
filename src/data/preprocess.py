import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_preprocessor(numeric_features, categorical_features, config):
    """Create preprocessing pipeline"""
    logger.info("Creating preprocessor")
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['preprocessing']['numeric_imputer_strategy'])),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(
            strategy=config['preprocessing']['categorical_imputer_strategy'],
            fill_value=config['preprocessing']['categorical_fill_value']
        )),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    logger.info(f"Preprocessor created with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    
    return preprocessor