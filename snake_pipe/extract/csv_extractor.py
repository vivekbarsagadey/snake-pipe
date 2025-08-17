"""
CSV data extractor
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CSVExtractor:
    """Extract data from CSV files"""
    
    def __init__(self, file_path: str, **kwargs):
        """
        Initialize CSV extractor
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional pandas read_csv parameters
        """
        self.file_path = Path(file_path)
        self.read_kwargs = kwargs
        
    def extract(self) -> pd.DataFrame:
        """
        Extract data from CSV file
        
        Returns:
            DataFrame containing the CSV data
        """
        try:
            logger.info(f"Extracting data from CSV: {self.file_path}")
            
            if not self.file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.file_path}")
            
            df = pd.read_csv(self.file_path, **self.read_kwargs)
            logger.info(f"Successfully extracted {len(df)} rows from CSV")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract data from CSV {self.file_path}: {str(e)}")
            raise
    
    def extract_sample(self, n_rows: int = 100) -> pd.DataFrame:
        """
        Extract a sample of data from CSV file
        
        Args:
            n_rows: Number of rows to sample
            
        Returns:
            DataFrame containing sample data
        """
        kwargs = self.read_kwargs.copy()
        kwargs['nrows'] = n_rows
        
        temp_extractor = CSVExtractor(self.file_path, **kwargs)
        return temp_extractor.extract()
    
    def get_column_info(self) -> Dict[str, Any]:
        """
        Get information about CSV columns
        
        Returns:
            Dictionary with column information
        """
        sample_df = self.extract_sample(10)
        
        return {
            'columns': list(sample_df.columns),
            'dtypes': sample_df.dtypes.to_dict(),
            'shape': sample_df.shape,
            'null_counts': sample_df.isnull().sum().to_dict()
        }