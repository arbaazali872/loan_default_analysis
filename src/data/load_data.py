import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw data from CSV file"""
    logger.info(f"Loading data from {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names"""
    logger.info("Cleaning column names")
    
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('.', '_')
        .str.replace('(', '')
        .str.replace(')', '')
        .str.replace('?', '')
        .str.replace('\'', '')
    )
    
    return df

def remove_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ID columns from dataframe"""
    logger.info("Removing ID columns")
    
    id_columns = [col for col in df.columns if '_id' in col]
    logger.info(f"Removing columns: {id_columns}")
    
    df = df.drop(columns=id_columns)
    
    return df