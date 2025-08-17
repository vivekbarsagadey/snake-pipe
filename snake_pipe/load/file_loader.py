"""
File data loader
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FileLoader:
    """Load data to various file formats"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize file loader
        
        Args:
            output_dir: Directory to save files to. If None, uses settings.DATA_DIR
        """
        self.output_dir = Path(output_dir) if output_dir else settings.DATA_DIR
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Initialized FileLoader with output directory: {self.output_dir}")
    
    def load_csv(self, df: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Load DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filename: Name of the CSV file
            **kwargs: Additional arguments for pandas to_csv
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.output_dir / filename
            if not filename.endswith('.csv'):
                file_path = self.output_dir / f"{filename}.csv"
            
            logger.info(f"Saving {len(df)} rows to CSV: {file_path}")
            
            # Default CSV parameters
            csv_kwargs = {
                'index': False,
                'encoding': 'utf-8'
            }
            csv_kwargs.update(kwargs)
            
            df.to_csv(file_path, **csv_kwargs)
            
            logger.info(f"Successfully saved CSV file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save CSV file {filename}: {str(e)}")
            raise
    
    def load_excel(self, df: pd.DataFrame, filename: str, sheet_name: str = 'Sheet1', 
                   **kwargs) -> str:
        """
        Load DataFrame to Excel file
        
        Args:
            df: DataFrame to save
            filename: Name of the Excel file
            sheet_name: Name of the sheet
            **kwargs: Additional arguments for pandas to_excel
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.output_dir / filename
            if not filename.endswith(('.xlsx', '.xls')):
                file_path = self.output_dir / f"{filename}.xlsx"
            
            logger.info(f"Saving {len(df)} rows to Excel: {file_path}")
            
            # Default Excel parameters
            excel_kwargs = {
                'index': False,
                'sheet_name': sheet_name
            }
            excel_kwargs.update(kwargs)
            
            df.to_excel(file_path, **excel_kwargs)
            
            logger.info(f"Successfully saved Excel file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save Excel file {filename}: {str(e)}")
            raise
    
    def load_parquet(self, df: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Load DataFrame to Parquet file
        
        Args:
            df: DataFrame to save
            filename: Name of the Parquet file
            **kwargs: Additional arguments for pandas to_parquet
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.output_dir / filename
            if not filename.endswith('.parquet'):
                file_path = self.output_dir / f"{filename}.parquet"
            
            logger.info(f"Saving {len(df)} rows to Parquet: {file_path}")
            
            # Default Parquet parameters
            parquet_kwargs = {
                'engine': 'pyarrow',
                'compression': 'snappy'
            }
            parquet_kwargs.update(kwargs)
            
            df.to_parquet(file_path, **parquet_kwargs)
            
            logger.info(f"Successfully saved Parquet file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save Parquet file {filename}: {str(e)}")
            raise
    
    def load_json(self, df: pd.DataFrame, filename: str, orient: str = 'records', 
                  **kwargs) -> str:
        """
        Load DataFrame to JSON file
        
        Args:
            df: DataFrame to save
            filename: Name of the JSON file
            orient: JSON orientation ('records', 'index', 'values', etc.)
            **kwargs: Additional arguments for pandas to_json
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.output_dir / filename
            if not filename.endswith('.json'):
                file_path = self.output_dir / f"{filename}.json"
            
            logger.info(f"Saving {len(df)} rows to JSON: {file_path}")
            
            # Default JSON parameters
            json_kwargs = {
                'orient': orient,
                'date_format': 'iso',
                'indent': 2
            }
            json_kwargs.update(kwargs)
            
            df.to_json(file_path, **json_kwargs)
            
            logger.info(f"Successfully saved JSON file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save JSON file {filename}: {str(e)}")
            raise
    
    def load_multiple_sheets(self, data: Dict[str, pd.DataFrame], filename: str) -> str:
        """
        Load multiple DataFrames to different sheets in an Excel file
        
        Args:
            data: Dictionary mapping sheet names to DataFrames
            filename: Name of the Excel file
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.output_dir / filename
            if not filename.endswith(('.xlsx', '.xls')):
                file_path = self.output_dir / f"{filename}.xlsx"
            
            logger.info(f"Saving {len(data)} sheets to Excel: {file_path}")
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"Added sheet '{sheet_name}' with {len(df)} rows")
            
            logger.info(f"Successfully saved multi-sheet Excel file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save multi-sheet Excel file {filename}: {str(e)}")
            raise
    
    def append_to_csv(self, df: pd.DataFrame, filename: str, **kwargs) -> str:
        """
        Append DataFrame to existing CSV file
        
        Args:
            df: DataFrame to append
            filename: Name of the CSV file
            **kwargs: Additional arguments for pandas to_csv
            
        Returns:
            Path to the file
        """
        try:
            file_path = self.output_dir / filename
            if not filename.endswith('.csv'):
                file_path = self.output_dir / f"{filename}.csv"
            
            logger.info(f"Appending {len(df)} rows to CSV: {file_path}")
            
            # Default CSV parameters for appending
            csv_kwargs = {
                'mode': 'a',
                'header': not file_path.exists(),  # Add header only if file doesn't exist
                'index': False,
                'encoding': 'utf-8'
            }
            csv_kwargs.update(kwargs)
            
            df.to_csv(file_path, **csv_kwargs)
            
            logger.info(f"Successfully appended to CSV file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to append to CSV file {filename}: {str(e)}")
            raise
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a saved file
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary with file information
        """
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        return {
            'filename': filename,
            'full_path': str(file_path),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'extension': file_path.suffix
        }
    
    def list_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List files in the output directory
        
        Args:
            pattern: Glob pattern to filter files
            
        Returns:
            List of file information dictionaries
        """
        files = []
        for file_path in self.output_dir.glob(pattern):
            if file_path.is_file():
                files.append(self.get_file_info(file_path.name))
        
        return sorted(files, key=lambda x: x['modified_time'], reverse=True)