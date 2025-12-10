"""
Preprocessing Utilities for Patient Data Analysis
==================================================

This module contains reusable functions for preprocessing patient data,
including missing value handling, encoding, feature engineering, and scaling.

Author: Data Science Team
Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, List, Optional


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

def handle_missing_values(df: pd.DataFrame, 
                         numeric_strategy: str = 'median',
                         categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_strategy : str, default='median'
        Strategy for numeric columns ('mean', 'median', 'mode')
    categorical_strategy : str, default='mode'
        Strategy for categorical columns ('mode', 'forward_fill', 'backward_fill')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if numeric_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numeric_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if categorical_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif categorical_strategy == 'forward_fill':
                df[col].fillna(method='ffill', inplace=True)
            elif categorical_strategy == 'backward_fill':
                df[col].fillna(method='bfill', inplace=True)
    
    return df


# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================

def encode_insurance_type(df: pd.DataFrame, 
                         column_name: str = 'insurance_type') -> pd.DataFrame:
    """
    Apply one-hot encoding to insurance type column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_name : str, default='insurance_type'
        Name of the insurance type column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded insurance types
    """
    df = df.copy()
    
    if column_name in df.columns:
        # One-hot encode
        encoded = pd.get_dummies(df[column_name], prefix='insurance', dtype=bool)
        
        # Drop original column and add encoded columns
        df = df.drop(columns=[column_name])
        df = pd.concat([df, encoded], axis=1)
    
    return df


def encode_categorical_columns(df: pd.DataFrame, 
                               columns: List[str],
                               drop_first: bool = True) -> pd.DataFrame:
    """
    Apply one-hot encoding to specified categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of column names to encode
    drop_first : bool, default=True
        Whether to drop the first category to avoid multicollinearity
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded columns
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            encoded = pd.get_dummies(df[col], prefix=col, drop_first=drop_first, dtype=bool)
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded], axis=1)
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_age_groups(df: pd.DataFrame, 
                     age_column: str = 'age') -> pd.DataFrame:
    """
    Create age group categories and one-hot encode them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    age_column : str, default='age'
        Name of the age column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with age group features
    """
    df = df.copy()
    
    # Define age groups
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['child', 'young_adult', 'middle_age', 'senior', 'elderly']
    
    # Create age group column
    df['age_group'] = pd.cut(df[age_column], bins=bins, labels=labels, right=False)
    
    # One-hot encode
    age_group_encoded = pd.get_dummies(df['age_group'], prefix='age_group', dtype=bool)
    df = pd.concat([df, age_group_encoded], axis=1)
    
    return df


def create_season_features(df: pd.DataFrame, 
                          month_column: str = 'arrival_month') -> pd.DataFrame:
    """
    Create season categories from month and one-hot encode them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    month_column : str, default='arrival_month'
        Name of the month column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with season features
    """
    df = df.copy()
    
    # Map months to seasons
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    
    # Create season column
    df['season'] = df[month_column].map(season_map)
    
    # One-hot encode
    season_encoded = pd.get_dummies(df['season'], prefix='season', dtype=bool)
    df = pd.concat([df, season_encoded], axis=1)
    
    return df


def create_stay_categories(df: pd.DataFrame, 
                          stay_column: str = 'length_of_stay') -> pd.DataFrame:
    """
    Create length of stay categories and one-hot encode them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    stay_column : str, default='length_of_stay'
        Name of the length of stay column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with stay category features
    """
    df = df.copy()
    
    # Define stay categories
    bins = [0, 4, 8, float('inf')]
    labels = ['short', 'medium', 'long']
    
    # Create stay category column
    df['stay_category'] = pd.cut(df[stay_column], bins=bins, labels=labels, right=False)
    
    # One-hot encode
    stay_encoded = pd.get_dummies(df['stay_category'], prefix='stay', dtype=bool)
    df = pd.concat([df, stay_encoded], axis=1)
    
    return df


def create_readmission_risk(df: pd.DataFrame,
                           prev_admissions_col: str = 'previous_admissions',
                           age_col: str = 'age',
                           length_of_stay_col: str = 'length_of_stay') -> pd.DataFrame:
    """
    Create a readmission risk score based on multiple factors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    prev_admissions_col : str
        Name of previous admissions column
    age_col : str
        Name of age column
    length_of_stay_col : str
        Name of length of stay column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with readmission risk score
    """
    df = df.copy()
    
    # Calculate risk score
    df['readmission_risk'] = (
        df[prev_admissions_col] * 0.4 +
        (df[age_col] / 100) * 0.3 +
        (df[length_of_stay_col] / 14) * 0.3
    )
    
    return df


# ============================================================================
# FEATURE SCALING
# ============================================================================

def apply_standard_scaling(df: pd.DataFrame, 
                          columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of columns to scale
    
    Returns:
    --------
    Tuple[pd.DataFrame, StandardScaler]
        Dataframe with scaled columns and fitted scaler
    """
    df = df.copy()
    scaler = StandardScaler()
    
    # Create new column names
    scaled_cols = [f"{col}_scaled" for col in columns]
    
    # Fit and transform
    scaled_values = scaler.fit_transform(df[columns])
    
    # Add scaled columns
    for i, col in enumerate(scaled_cols):
        df[col] = scaled_values[:, i]
    
    return df, scaler


def apply_minmax_scaling(df: pd.DataFrame, 
                        columns: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMaxScaler to specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of columns to scale
    
    Returns:
    --------
    Tuple[pd.DataFrame, MinMaxScaler]
        Dataframe with scaled columns and fitted scaler
    """
    df = df.copy()
    scaler = MinMaxScaler()
    
    # Create new column names
    scaled_cols = [f"{col}_scaled" for col in columns]
    
    # Fit and transform
    scaled_values = scaler.fit_transform(df[columns])
    
    # Add scaled columns
    for i, col in enumerate(scaled_cols):
        df[col] = scaled_values[:, i]
    
    return df, scaler


def apply_robust_scaling(df: pd.DataFrame, 
                        columns: List[str]) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Apply RobustScaler to specified columns (robust to outliers).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of columns to scale
    
    Returns:
    --------
    Tuple[pd.DataFrame, RobustScaler]
        Dataframe with scaled columns and fitted scaler
    """
    df = df.copy()
    scaler = RobustScaler()
    
    # Create new column names
    scaled_cols = [f"{col}_scaled" for col in columns]
    
    # Fit and transform
    scaled_values = scaler.fit_transform(df[columns])
    
    # Add scaled columns
    for i, col in enumerate(scaled_cols):
        df[col] = scaled_values[:, i]
    
    return df, scaler


# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_patient_data(df: pd.DataFrame,
                           handle_missing: bool = True,
                           encode_categories: bool = True,
                           engineer_features: bool = True,
                           scale_features: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for patient data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    handle_missing : bool, default=True
        Whether to handle missing values
    encode_categories : bool, default=True
        Whether to encode categorical variables
    engineer_features : bool, default=True
        Whether to create engineered features
    scale_features : bool, default=True
        Whether to scale numeric features
    
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed dataframe
    """
    df = df.copy()
    
    print("🔄 Starting preprocessing pipeline...")
    
    # Step 1: Handle missing values
    if handle_missing:
        print("  ✓ Handling missing values...")
        df = handle_missing_values(df)
    
    # Step 2: Encode categorical variables
    if encode_categories:
        print("  ✓ Encoding categorical variables...")
        if 'insurance_type' in df.columns:
            df = encode_insurance_type(df)
    
    # Step 3: Feature engineering
    if engineer_features:
        print("  ✓ Engineering new features...")
        if 'age' in df.columns:
            df = create_age_groups(df)
        if 'arrival_month' in df.columns:
            df = create_season_features(df)
        if 'length_of_stay' in df.columns:
            df = create_stay_categories(df)
        if all(col in df.columns for col in ['previous_admissions', 'age', 'length_of_stay']):
            df = create_readmission_risk(df)
    
    # Step 4: Scale features
    if scale_features:
        print("  ✓ Scaling features...")
        if 'age' in df.columns and 'satisfaction' in df.columns:
            df, _ = apply_standard_scaling(df, ['age', 'satisfaction'])
        if 'arrival_month' in df.columns:
            df, _ = apply_minmax_scaling(df, ['arrival_month'])
        if 'length_of_stay' in df.columns:
            df, _ = apply_robust_scaling(df, ['length_of_stay'])
    
    print("✅ Preprocessing complete!")
    
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_preprocessing_summary(df_original: pd.DataFrame, 
                             df_processed: pd.DataFrame) -> dict:
    """
    Generate a summary of preprocessing changes.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe
    df_processed : pd.DataFrame
        Processed dataframe
    
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'original_shape': df_original.shape,
        'processed_shape': df_processed.shape,
        'features_added': df_processed.shape[1] - df_original.shape[1],
        'missing_values_original': df_original.isnull().sum().sum(),
        'missing_values_processed': df_processed.isnull().sum().sum()
    }
    
    return summary


if __name__ == "__main__":
    print("Preprocessing Utils Module - Ready to use!")
    print("\nAvailable functions:")
    print("  • handle_missing_values()")
    print("  • encode_insurance_type()")
    print("  • create_age_groups()")
    print("  • create_season_features()")
    print("  • create_stay_categories()")
    print("  • apply_standard_scaling()")
    print("  • apply_minmax_scaling()")
    print("  • apply_robust_scaling()")
    print("  • preprocess_patient_data() - Complete pipeline")
