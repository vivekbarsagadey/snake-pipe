"""
Data cleaning utilities
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean and standardize data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data cleaner
        
        Args:
            df: DataFrame to clean
        """
        self.df = df.copy()
        self.original_shape = df.shape
        logger.info(f"Initialized DataCleaner with {self.original_shape[0]} rows, {self.original_shape[1]} columns")
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows
        
        Args:
            subset: Column names to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Self for method chaining
        """
        before_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after_count = len(self.df)
        
        logger.info(f"Removed {before_count - after_count} duplicate rows")
        return self
    
    def handle_missing_values(self, strategy: str = 'drop', columns: Optional[List[str]] = None,
                            fill_value: Any = None) -> 'DataCleaner':
        """
        Handle missing values
        
        Args:
            strategy: How to handle missing values ('drop', 'fill', 'forward_fill', 'backward_fill')
            columns: Specific columns to process (None for all)
            fill_value: Value to use for filling (required for 'fill' strategy)
            
        Returns:
            Self for method chaining
        """
        target_cols = columns or self.df.columns.tolist()
        missing_before = self.df[target_cols].isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna(subset=target_cols)
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value is required for 'fill' strategy")
            self.df[target_cols] = self.df[target_cols].fillna(fill_value)
        elif strategy == 'forward_fill':
            self.df[target_cols] = self.df[target_cols].fillna(method='ffill')
        elif strategy == 'backward_fill':
            self.df[target_cols] = self.df[target_cols].fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        missing_after = self.df[target_cols].isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values using {strategy} strategy")
        return self
    
    def standardize_columns(self, case: str = 'lower', replace_spaces: str = '_') -> 'DataCleaner':
        """
        Standardize column names
        
        Args:
            case: Case to convert to ('lower', 'upper', 'title')
            replace_spaces: Character to replace spaces with
            
        Returns:
            Self for method chaining
        """
        old_columns = self.df.columns.tolist()
        
        new_columns = []
        for col in old_columns:
            new_col = str(col).replace(' ', replace_spaces)
            if case == 'lower':
                new_col = new_col.lower()
            elif case == 'upper':
                new_col = new_col.upper()
            elif case == 'title':
                new_col = new_col.title()
            new_columns.append(new_col)
        
        self.df.columns = new_columns
        logger.info(f"Standardized column names: {case} case, spaces replaced with '{replace_spaces}'")
        return self
    
    def remove_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        """
        Remove outliers from specified columns
        
        Args:
            columns: List of numeric columns to check for outliers
            method: Method to use ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Self for method chaining
        """
        before_count = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
                
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column {col} is not numeric, skipping")
                continue
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                self.df = self.df[mask]
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                mask = z_scores <= threshold
                self.df = self.df[mask]
        
        after_count = len(self.df)
        logger.info(f"Removed {before_count - after_count} outlier rows using {method} method")
        return self
    
    def apply_custom_transform(self, func: Callable[[pd.DataFrame], pd.DataFrame], 
                             description: str = "custom transform") -> 'DataCleaner':
        """
        Apply a custom transformation function
        
        Args:
            func: Function that takes and returns a DataFrame
            description: Description of the transformation
            
        Returns:
            Self for method chaining
        """
        before_shape = self.df.shape
        self.df = func(self.df)
        after_shape = self.df.shape
        
        logger.info(f"Applied {description}: shape changed from {before_shape} to {after_shape}")
        return self
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        final_shape = self.df.shape
        logger.info(f"Cleaning complete: {self.original_shape} -> {final_shape}")
        return self.df
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the cleaning operations
        
        Returns:
            Dictionary with cleaning summary
        """
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_count': self.df.shape[1],
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'dtypes': self.df.dtypes.to_dict()
        }