"""
Main ETL pipeline orchestration
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from .extract import APIExtractor, CSVExtractor, DatabaseExtractor
from .load import DatabaseLoader, FileLoader
from .transform import DataCleaner, DataEnricher, DataValidator
from .utils.logger import get_logger

logger = get_logger(__name__)


class Pipeline:
    """Main ETL Pipeline class"""

    def __init__(self, name: str = "snake_pipe_pipeline"):
        """
        Initialize pipeline

        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.steps: List[Tuple[str, Callable[[], None]]] = []
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        logger.info(f"Initialized pipeline: {name}")

    def extract_from_csv(self, file_path: str, **kwargs: Any) -> "Pipeline":
        """
        Add CSV extraction step

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for CSV extraction

        Returns:
            Self for method chaining
        """

        def extract_step() -> None:
            extractor = CSVExtractor(file_path, **kwargs)
            self.data = extractor.extract()
            self.metadata["source"] = f"CSV: {file_path}"
            if self.data is not None:
                self.metadata["rows_extracted"] = len(self.data)
                logger.info(f"Extracted {len(self.data)} rows from CSV: {file_path}")

        self.steps.append(("extract_csv", extract_step))
        return self

    def extract_from_database(self, query: str, connection_string: Optional[str] = None, **kwargs: Any) -> "Pipeline":
        """
        Add database extraction step

        Args:
            query: SQL query to execute
            connection_string: Database connection string
            **kwargs: Additional arguments for database extraction

        Returns:
            Self for method chaining
        """

        def extract_step() -> None:
            extractor = DatabaseExtractor(connection_string)
            self.data = extractor.extract_query(query, **kwargs)
            self.metadata["source"] = f"Database: {query[:50]}..."
            if self.data is not None:
                self.metadata["rows_extracted"] = len(self.data)
                logger.info(f"Extracted {len(self.data)} rows from database")

        self.steps.append(("extract_database", extract_step))
        return self

    def extract_from_api(self, endpoint: str, base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any) -> "Pipeline":
        """
        Add API extraction step

        Args:
            endpoint: API endpoint to call
            base_url: Base URL for API
            api_key: API key for authentication
            **kwargs: Additional arguments for API extraction

        Returns:
            Self for method chaining
        """

        def extract_step() -> None:
            extractor = APIExtractor(base_url, api_key)
            data = extractor.extract_endpoint(endpoint, **kwargs)
            self.data = extractor.to_dataframe(data)
            self.metadata["source"] = f"API: {endpoint}"
            if self.data is not None:
                self.metadata["rows_extracted"] = len(self.data)
                logger.info(f"Extracted {len(self.data)} rows from API: {endpoint}")

        self.steps.append(("extract_api", extract_step))
        return self

    def clean_data(self, **kwargs: Any) -> "Pipeline":
        """
        Add data cleaning step

        Args:
            **kwargs: Arguments for DataCleaner methods

        Returns:
            Self for method chaining
        """

        def clean_step() -> None:
            if self.data is None:
                raise ValueError("No data to clean. Run an extraction step first.")

            cleaner = DataCleaner(self.data)

            # Apply default cleaning operations
            cleaner.remove_duplicates()
            cleaner.standardize_columns()

            # Apply custom cleaning if specified
            for method_name, method_kwargs in kwargs.items():
                if hasattr(cleaner, method_name):
                    method = getattr(cleaner, method_name)
                    if isinstance(method_kwargs, dict):
                        method(**method_kwargs)
                    else:
                        method(method_kwargs)

            self.data = cleaner.get_data()
            if self.data is not None:
                self.metadata["rows_after_cleaning"] = len(self.data)
                logger.info(f"Data cleaned: {len(self.data)} rows remaining")

        self.steps.append(("clean_data", clean_step))
        return self

    def validate_data(self, **kwargs: Any) -> "Pipeline":
        """
        Add data validation step

        Args:
            **kwargs: Arguments for DataValidator methods

        Returns:
            Self for method chaining
        """

        def validate_step() -> None:
            if self.data is None:
                raise ValueError("No data to validate. Run an extraction step first.")

            validator = DataValidator(self.data)

            # Apply validation rules
            for method_name, method_kwargs in kwargs.items():
                if hasattr(validator, method_name):
                    method = getattr(validator, method_name)
                    if isinstance(method_kwargs, dict):
                        method(**method_kwargs)
                    else:
                        method(method_kwargs)

            result = validator.get_result()
            self.metadata["validation_result"] = result.summary()

            if not result.is_valid:
                error_msg = f"Data validation failed: {len(result.errors)} errors found"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Data validation passed")

        self.steps.append(("validate_data", validate_step))
        return self

    def enrich_data(self, **kwargs: Any) -> "Pipeline":
        """
        Add data enrichment step

        Args:
            **kwargs: Arguments for DataEnricher methods

        Returns:
            Self for method chaining
        """

        def enrich_step() -> None:
            if self.data is None:
                raise ValueError("No data to enrich. Run an extraction step first.")

            enricher = DataEnricher(self.data)

            # Apply enrichment operations
            for method_name, method_kwargs in kwargs.items():
                if hasattr(enricher, method_name):
                    method = getattr(enricher, method_name)
                    if isinstance(method_kwargs, dict):
                        method(**method_kwargs)
                    else:
                        method(method_kwargs)

            self.data = enricher.get_data()
            if self.data is not None:
                self.metadata["columns_after_enrichment"] = len(self.data.columns)
                logger.info(f"Data enriched: {len(self.data.columns)} columns")

        self.steps.append(("enrich_data", enrich_step))
        return self

    def load_to_csv(self, filename: str, output_dir: Optional[str] = None, **kwargs: Any) -> "Pipeline":
        """
        Add CSV loading step

        Args:
            filename: Output filename
            output_dir: Output directory
            **kwargs: Additional arguments for CSV loading

        Returns:
            Self for method chaining
        """

        def load_step() -> None:
            if self.data is None:
                raise ValueError("No data to load. Run extraction and transformation steps first.")

            loader = FileLoader(output_dir)
            file_path = loader.load_csv(self.data, filename, **kwargs)
            self.metadata["output_file"] = file_path
            logger.info(f"Data loaded to CSV: {file_path}")

        self.steps.append(("load_csv", load_step))
        return self

    def load_to_database(self, table_name: str, connection_string: Optional[str] = None, **kwargs: Any) -> "Pipeline":
        """
        Add database loading step

        Args:
            table_name: Target table name
            connection_string: Database connection string
            **kwargs: Additional arguments for database loading

        Returns:
            Self for method chaining
        """

        def load_step() -> None:
            if self.data is None:
                raise ValueError("No data to load. Run extraction and transformation steps first.")

            loader = DatabaseLoader(connection_string)
            loader.load_dataframe(self.data, table_name, **kwargs)
            self.metadata["output_table"] = table_name
            logger.info(f"Data loaded to database table: {table_name}")

        self.steps.append(("load_database", load_step))
        return self

    def apply_custom_transform(self, transform_func: Callable[[pd.DataFrame], pd.DataFrame], description: str = "custom transform") -> "Pipeline":
        """
        Add custom transformation step

        Args:
            transform_func: Function that takes and returns a DataFrame
            description: Description of the transformation

        Returns:
            Self for method chaining
        """

        def transform_step() -> None:
            if self.data is None:
                raise ValueError("No data to transform. Run an extraction step first.")

            self.data = transform_func(self.data)
            if self.data is not None:
                logger.info(f"Applied {description}: {len(self.data)} rows")

        self.steps.append((f"custom_transform_{description}", transform_step))
        return self

    def run(self) -> pd.DataFrame:
        """
        Execute all pipeline steps

        Returns:
            Final processed DataFrame
        """
        logger.info(f"Starting pipeline execution: {self.name}")
        logger.info(f"Pipeline has {len(self.steps)} steps")

        try:
            for i, (step_name, step_func) in enumerate(self.steps, 1):
                logger.info(f"Executing step {i}/{len(self.steps)}: {step_name}")
                step_func()

            logger.info(f"Pipeline '{self.name}' completed successfully")
            if self.data is not None:
                logger.info(f"Final data shape: {self.data.shape}")
                return self.data
            else:
                logger.warning("Pipeline completed but no data was generated")
                return pd.DataFrame()  # Return empty DataFrame instead of None

        except Exception as e:
            logger.error(f"Pipeline '{self.name}' failed: {str(e)}")
            raise

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get pipeline metadata

        Returns:
            Dictionary with pipeline metadata
        """
        return {
            "pipeline_name": self.name,
            "steps_count": len(self.steps),
            "steps": [step_name for step_name, _ in self.steps],
            "final_shape": self.data.shape if self.data is not None else None,
            **self.metadata,
        }

    def preview(self, n_rows: int = 5) -> Optional[pd.DataFrame]:
        """
        Preview the current data

        Args:
            n_rows: Number of rows to preview

        Returns:
            Preview DataFrame or None if no data
        """
        if self.data is not None:
            return self.data.head(n_rows)
        return None


def run_pipeline() -> None:
    """
    Example pipeline execution function
    This is the main entry point for the CLI
    """
    logger.info("Starting snake-pipe ETL pipeline")

    # Example pipeline - replace with your actual pipeline logic
    pipeline = Pipeline("example_pipeline")

    try:
        # Example: Extract from CSV, clean, and load to another CSV
        result = pipeline.extract_from_csv("sample_data.csv").clean_data().load_to_csv("processed_data.csv").run()

        logger.info(f"Pipeline completed successfully. Processed {len(result)} rows.")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_pipeline()
