"""
Database data extractor
"""

from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sqlalchemy import create_engine, text
except ImportError:
    create_engine = text = None

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseExtractor:
    """Extract data from databases using SQLAlchemy"""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database extractor

        Args:
            connection_string: Database connection string. If None, uses settings.DATABASE_URL
        """
        self.connection_string = connection_string or settings.DATABASE_URL
        if not self.connection_string:
            raise ValueError("Database connection string is required")

        self.engine = create_engine(self.connection_string)

    def extract_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Extract data using a SQL query

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            DataFrame containing query results
        """
        try:
            logger.info(f"Executing query: {query[:100]}...")

            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)

            logger.info(f"Successfully extracted {len(df)} rows from database")
            return df

        except Exception as e:
            logger.error(f"Failed to extract data from database: {str(e)}")
            raise

    def extract_table(self, table_name: str, schema: Optional[str] = None, columns: Optional[List[str]] = None, where_clause: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extract data from a specific table

        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            columns: List of columns to select (optional, defaults to all)
            where_clause: WHERE clause condition (optional)
            limit: Limit number of rows (optional)

        Returns:
            DataFrame containing table data
        """
        # Build the query safely
        cols = ", ".join(columns) if columns else "*"

        # Safely quote identifiers to prevent SQL injection
        if hasattr(self.engine, "dialect") and hasattr(self.engine.dialect, "identifier_preparer"):
            preparer = self.engine.dialect.identifier_preparer
            quoted_table = preparer.quote(table_name)
            if schema:
                quoted_schema = preparer.quote(schema)
                table_ref = f"{quoted_schema}.{quoted_table}"
            else:
                table_ref = quoted_table
        else:
            # Fallback: basic validation and quoting
            if table_name and not table_name.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid table name")
            if schema and not schema.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid schema name")
            table_ref = f"{schema}.{table_name}" if schema else table_name

        query = f"SELECT {cols} FROM {table_ref}"  # nosec B608

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        return self.extract_query(query)

    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a table

        Args:
            table_name: Name of the table
            schema: Schema name (optional)

        Returns:
            Dictionary with table information
        """
        table_ref = f"{schema}.{table_name}" if schema else table_name

        # Get column information using parameterized query
        from sqlalchemy import text

        info_query = text(
            """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = :table_name
        AND (:schema IS NULL OR table_schema = :schema)
        """
        )

        column_info = self.extract_query(info_query, {"table_name": table_name, "schema": schema})

        # Get row count using safely quoted table name
        if hasattr(self.engine, "dialect") and hasattr(self.engine.dialect, "identifier_preparer"):
            preparer = self.engine.dialect.identifier_preparer
            quoted_table = preparer.quote(table_name)
            if schema:
                quoted_schema = preparer.quote(schema)
                table_ref = f"{quoted_schema}.{quoted_table}"
            else:
                table_ref = quoted_table
        else:
            # Fallback: basic validation
            if not table_name.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid table name")
            if schema and not schema.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid schema name")
            table_ref = f"{schema}.{table_name}" if schema else table_name

        count_query = f"SELECT COUNT(*) as row_count FROM {table_ref}"  # nosec B608
        row_count = self.extract_query(count_query)["row_count"].iloc[0]

        return {"table_name": table_name, "schema": schema, "columns": column_info.to_dict("records"), "row_count": row_count}

    def close(self) -> None:
        """Close database connection"""
        if hasattr(self, "engine"):
            self.engine.dispose()
