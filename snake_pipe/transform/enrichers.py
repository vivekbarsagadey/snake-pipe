"""
Data enrichment utilities
"""

from typing import Any, Callable, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataEnricher:
    """Enrich data with additional computed fields and transformations"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize data enricher

        Args:
            df: DataFrame to enrich
        """
        self.df = df.copy()
        logger.info(f"Initialized DataEnricher with {len(df)} rows, {len(df.columns)} columns")

    def add_calculated_column(self, column_name: str, calculation: Callable[[pd.DataFrame], pd.Series], description: str = "") -> "DataEnricher":
        """
        Add a calculated column based on existing data

        Args:
            column_name: Name of the new column
            calculation: Function that takes DataFrame and returns Series
            description: Description of the calculation

        Returns:
            Self for method chaining
        """
        try:
            self.df[column_name] = calculation(self.df)
            logger.info(f"Added calculated column '{column_name}': {description}")
        except Exception as e:
            logger.error(f"Failed to add calculated column '{column_name}': {str(e)}")
            raise

        return self

    def add_datetime_features(self, datetime_column: str, features: Optional[List[str]] = None) -> "DataEnricher":
        """
        Extract datetime features from a datetime column

        Args:
            datetime_column: Name of the datetime column
            features: List of features to extract. Options: 'year', 'month', 'day',
                     'hour', 'minute', 'weekday', 'quarter', 'week_of_year', 'is_weekend'

        Returns:
            Self for method chaining
        """
        if datetime_column not in self.df.columns:
            raise ValueError(f"Column {datetime_column} not found")

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_column]):
            self.df[datetime_column] = pd.to_datetime(self.df[datetime_column])

        if features is None:
            features = ["year", "month", "day", "hour", "weekday", "quarter"]

        dt_series = self.df[datetime_column]

        for feature in features:
            if feature == "year":
                self.df[f"{datetime_column}_year"] = dt_series.dt.year
            elif feature == "month":
                self.df[f"{datetime_column}_month"] = dt_series.dt.month
            elif feature == "day":
                self.df[f"{datetime_column}_day"] = dt_series.dt.day
            elif feature == "hour":
                self.df[f"{datetime_column}_hour"] = dt_series.dt.hour
            elif feature == "minute":
                self.df[f"{datetime_column}_minute"] = dt_series.dt.minute
            elif feature == "weekday":
                self.df[f"{datetime_column}_weekday"] = dt_series.dt.dayofweek
            elif feature == "quarter":
                self.df[f"{datetime_column}_quarter"] = dt_series.dt.quarter
            elif feature == "week_of_year":
                self.df[f"{datetime_column}_week"] = dt_series.dt.isocalendar().week
            elif feature == "is_weekend":
                self.df[f"{datetime_column}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6])

        logger.info(f"Added datetime features for {datetime_column}: {features}")
        return self

    def add_categorical_encoding(self, categorical_columns: List[str], encoding_type: str = "label") -> "DataEnricher":
        """
        Add encoded versions of categorical columns

        Args:
            categorical_columns: List of categorical column names
            encoding_type: Type of encoding ('label', 'onehot', 'target')

        Returns:
            Self for method chaining
        """
        for column in categorical_columns:
            if column not in self.df.columns:
                logger.warning(f"Column {column} not found, skipping")
                continue

            if encoding_type == "label":
                # Simple label encoding
                unique_values = self.df[column].unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self.df[f"{column}_encoded"] = self.df[column].map(mapping)

            elif encoding_type == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(self.df[column], prefix=column)
                self.df = pd.concat([self.df, dummies], axis=1)

        logger.info(f"Added {encoding_type} encoding for columns: {categorical_columns}")
        return self

    def add_aggregated_features(self, group_by_columns: List[str], agg_column: str, agg_functions: List[str]) -> "DataEnricher":
        """
        Add aggregated features based on grouping

        Args:
            group_by_columns: Columns to group by
            agg_column: Column to aggregate
            agg_functions: List of aggregation functions ('mean', 'sum', 'count', 'std', etc.)

        Returns:
            Self for method chaining
        """
        for func in agg_functions:
            try:
                agg_data = self.df.groupby(group_by_columns)[agg_column].agg(func).reset_index()

                # Create new column name
                group_str = "_".join(group_by_columns)
                new_column = f"{agg_column}_{func}_by_{group_str}"

                # Merge back to original dataframe
                self.df = self.df.merge(agg_data.rename(columns={agg_column: new_column}), on=group_by_columns, how="left")

            except Exception as e:
                logger.error(f"Failed to add aggregated feature {func} for {agg_column}: {str(e)}")

        logger.info(f"Added aggregated features for {agg_column} grouped by {group_by_columns}")
        return self

    def add_text_features(self, text_column: str) -> "DataEnricher":
        """
        Add text-based features from a text column

        Args:
            text_column: Name of the text column

        Returns:
            Self for method chaining
        """
        if text_column not in self.df.columns:
            raise ValueError(f"Column {text_column} not found")

        text_series = self.df[text_column].astype(str)

        # Basic text features
        self.df[f"{text_column}_length"] = text_series.str.len()
        self.df[f"{text_column}_word_count"] = text_series.str.split().str.len()
        self.df[f"{text_column}_char_count"] = text_series.str.len()
        self.df[f"{text_column}_uppercase_ratio"] = (text_series.str.count(r"[A-Z]") / text_series.str.len()).fillna(0)

        logger.info(f"Added text features for column: {text_column}")
        return self

    def add_binned_features(self, numeric_columns: List[str], bins: int = 5, labels: Optional[List[str]] = None) -> "DataEnricher":
        """
        Add binned versions of numeric columns

        Args:
            numeric_columns: List of numeric column names
            bins: Number of bins or list of bin edges
            labels: Labels for the bins

        Returns:
            Self for method chaining
        """
        for column in numeric_columns:
            if column not in self.df.columns:
                logger.warning(f"Column {column} not found, skipping")
                continue

            if not pd.api.types.is_numeric_dtype(self.df[column]):
                logger.warning(f"Column {column} is not numeric, skipping")
                continue

            try:
                self.df[f"{column}_binned"] = pd.cut(self.df[column], bins=bins, labels=labels, duplicates="drop")
            except Exception as e:
                logger.error(f"Failed to bin column {column}: {str(e)}")

        logger.info(f"Added binned features for columns: {numeric_columns}")
        return self

    def add_flag_columns(self, conditions: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> "DataEnricher":
        """
        Add boolean flag columns based on conditions

        Args:
            conditions: Dictionary mapping flag names to condition functions

        Returns:
            Self for method chaining
        """
        for flag_name, condition in conditions.items():
            try:
                self.df[flag_name] = condition(self.df)
                logger.info(f"Added flag column: {flag_name}")
            except Exception as e:
                logger.error(f"Failed to add flag column {flag_name}: {str(e)}")

        return self

    def get_data(self) -> pd.DataFrame:
        """
        Get the enriched DataFrame

        Returns:
            Enriched DataFrame
        """
        logger.info(f"Enrichment complete: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the enrichment operations

        Returns:
            Dictionary with enrichment summary
        """
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "column_names": list(self.df.columns),
            "memory_usage": self.df.memory_usage(deep=True).sum(),
            "dtypes": self.df.dtypes.to_dict(),
        }
