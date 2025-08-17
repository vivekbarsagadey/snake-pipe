"""
Data validation utilities
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Union
from pydantic import BaseModel, ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Container for validation results"""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def add_error(self, message: str, column: Optional[str] = None, row_index: Optional[int] = None):
        """Add an error to the validation result"""
        self.is_valid = False
        error = {'message': message, 'column': column, 'row_index': row_index}
        self.errors.append(error)
    
    def add_warning(self, message: str, column: Optional[str] = None):
        """Add a warning to the validation result"""
        warning = {'message': message, 'column': column}
        self.warnings.append(warning)
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of validation results"""
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats
        }


class DataValidator:
    """Validate data quality and constraints"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data validator
        
        Args:
            df: DataFrame to validate
        """
        self.df = df
        self.result = ValidationResult()
        logger.info(f"Initialized DataValidator with {len(df)} rows, {len(df.columns)} columns")
    
    def check_required_columns(self, required_columns: List[str]) -> 'DataValidator':
        """
        Check if required columns exist
        
        Args:
            required_columns: List of column names that must exist
            
        Returns:
            Self for method chaining
        """
        missing_columns = set(required_columns) - set(self.df.columns)
        
        if missing_columns:
            self.result.add_error(f"Missing required columns: {list(missing_columns)}")
        else:
            logger.info("All required columns present")
        
        return self
    
    def check_data_types(self, expected_types: Dict[str, str]) -> 'DataValidator':
        """
        Check if columns have expected data types
        
        Args:
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            Self for method chaining
        """
        for column, expected_type in expected_types.items():
            if column not in self.df.columns:
                self.result.add_warning(f"Column {column} not found for type checking")
                continue
            
            actual_type = str(self.df[column].dtype)
            
            # Flexible type checking
            type_mapping = {
                'int': ['int64', 'int32', 'int16', 'int8'],
                'float': ['float64', 'float32'],
                'string': ['object', 'string'],
                'datetime': ['datetime64[ns]', 'datetime64'],
                'bool': ['bool']
            }
            
            expected_dtypes = type_mapping.get(expected_type, [expected_type])
            
            if actual_type not in expected_dtypes:
                self.result.add_error(
                    f"Column {column} has type {actual_type}, expected {expected_type}",
                    column=column
                )
        
        return self
    
    def check_null_values(self, columns: Optional[List[str]] = None, 
                         max_null_percentage: float = 0.0) -> 'DataValidator':
        """
        Check for null values in specified columns
        
        Args:
            columns: List of columns to check (None for all)
            max_null_percentage: Maximum allowed percentage of null values
            
        Returns:
            Self for method chaining
        """
        check_columns = columns or self.df.columns.tolist()
        
        for column in check_columns:
            if column not in self.df.columns:
                continue
            
            null_count = self.df[column].isnull().sum()
            null_percentage = (null_count / len(self.df)) * 100
            
            if null_percentage > max_null_percentage:
                self.result.add_error(
                    f"Column {column} has {null_percentage:.2f}% null values "
                    f"(max allowed: {max_null_percentage}%)",
                    column=column
                )
        
        return self
    
    def check_unique_values(self, columns: List[str]) -> 'DataValidator':
        """
        Check if specified columns have unique values
        
        Args:
            columns: List of columns that should have unique values
            
        Returns:
            Self for method chaining
        """
        for column in columns:
            if column not in self.df.columns:
                self.result.add_warning(f"Column {column} not found for uniqueness check")
                continue
            
            duplicate_count = self.df[column].duplicated().sum()
            
            if duplicate_count > 0:
                self.result.add_error(
                    f"Column {column} has {duplicate_count} duplicate values",
                    column=column
                )
        
        return self
    
    def check_value_ranges(self, range_constraints: Dict[str, Dict[str, Union[int, float]]]) -> 'DataValidator':
        """
        Check if numeric columns fall within specified ranges
        
        Args:
            range_constraints: Dictionary mapping column names to {'min': value, 'max': value}
            
        Returns:
            Self for method chaining
        """
        for column, constraints in range_constraints.items():
            if column not in self.df.columns:
                self.result.add_warning(f"Column {column} not found for range check")
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                self.result.add_warning(f"Column {column} is not numeric, skipping range check")
                continue
            
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None:
                below_min = (self.df[column] < min_val).sum()
                if below_min > 0:
                    self.result.add_error(
                        f"Column {column} has {below_min} values below minimum {min_val}",
                        column=column
                    )
            
            if max_val is not None:
                above_max = (self.df[column] > max_val).sum()
                if above_max > 0:
                    self.result.add_error(
                        f"Column {column} has {above_max} values above maximum {max_val}",
                        column=column
                    )
        
        return self
    
    def check_allowed_values(self, value_constraints: Dict[str, List[Any]]) -> 'DataValidator':
        """
        Check if columns contain only allowed values
        
        Args:
            value_constraints: Dictionary mapping column names to lists of allowed values
            
        Returns:
            Self for method chaining
        """
        for column, allowed_values in value_constraints.items():
            if column not in self.df.columns:
                self.result.add_warning(f"Column {column} not found for value check")
                continue
            
            invalid_values = set(self.df[column].unique()) - set(allowed_values)
            
            if invalid_values:
                self.result.add_error(
                    f"Column {column} contains invalid values: {list(invalid_values)}",
                    column=column
                )
        
        return self
    
    def apply_custom_validation(self, func: Callable[[pd.DataFrame], List[str]], 
                              description: str = "custom validation") -> 'DataValidator':
        """
        Apply a custom validation function
        
        Args:
            func: Function that takes DataFrame and returns list of error messages
            description: Description of the validation
            
        Returns:
            Self for method chaining
        """
        try:
            errors = func(self.df)
            for error in errors:
                self.result.add_error(f"{description}: {error}")
            
            if not errors:
                logger.info(f"Custom validation '{description}' passed")
        
        except Exception as e:
            self.result.add_error(f"Custom validation '{description}' failed: {str(e)}")
        
        return self
    
    def get_result(self) -> ValidationResult:
        """
        Get the validation result
        
        Returns:
            ValidationResult object
        """
        # Add summary stats
        self.result.stats = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'null_percentages': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
        
        logger.info(f"Validation complete: {'PASSED' if self.result.is_valid else 'FAILED'}")
        return self.result