"""
Database data loader
"""

from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sqlalchemy import MetaData, Table, create_engine
    from sqlalchemy.dialects.postgresql import insert
except ImportError:
    MetaData = Table = create_engine = insert = None

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """Load data to databases using SQLAlchemy"""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database loader

        Args:
            connection_string: Database connection string. If None, uses settings.DATABASE_URL
        """
        self.connection_string = connection_string or settings.DATABASE_URL
        if not self.connection_string:
            raise ValueError("Database connection string is required")

        self.engine = create_engine(self.connection_string)
        self.metadata = MetaData()

    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace", index: bool = False, schema: Optional[str] = None, batch_size: Optional[int] = None) -> None:
        """
        Load DataFrame to database table

        Args:
            df: DataFrame to load
            table_name: Name of the target table
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            index: Whether to write DataFrame index as a column
            schema: Schema name
            batch_size: Number of rows to insert in each batch
        """
        try:
            logger.info(f"Loading {len(df)} rows to table {table_name}")

            batch_size = batch_size or settings.BATCH_SIZE

            # Load data in batches if batch_size is specified
            if batch_size and len(df) > batch_size:
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i: i + batch_size]
                    batch_if_exists = "append" if i > 0 else if_exists

                    batch_df.to_sql(name=table_name, con=self.engine, if_exists=batch_if_exists, index=index, schema=schema)

                    logger.info(f"Loaded batch {i//batch_size + 1}: {len(batch_df)} rows")
            else:
                df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=index, schema=schema)

            logger.info(f"Successfully loaded {len(df)} rows to {table_name}")

        except Exception as e:
            logger.error(f"Failed to load data to {table_name}: {str(e)}")
            raise

    def upsert_dataframe(self, df: pd.DataFrame, table_name: str, primary_keys: List[str], schema: Optional[str] = None) -> None:
        """
        Upsert DataFrame to database table (PostgreSQL specific)

        Args:
            df: DataFrame to upsert
            table_name: Name of the target table
            primary_keys: List of column names that form the primary key
            schema: Schema name
        """
        try:
            logger.info(f"Upserting {len(df)} rows to table {table_name}")

            # Reflect the existing table structure
            table = Table(table_name, self.metadata, autoload_with=self.engine, schema=schema)

            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    # Convert row to dictionary
                    row_dict = row.to_dict()

                    # Create insert statement with ON CONFLICT DO UPDATE
                    stmt = insert(table).values(row_dict)

                    # Build update dictionary (exclude primary keys)
                    update_dict = {col.name: stmt.excluded[col.name] for col in table.columns if col.name not in primary_keys}

                    if update_dict:
                        stmt = stmt.on_conflict_do_update(index_elements=primary_keys, set_=update_dict)
                    else:
                        stmt = stmt.on_conflict_do_nothing(index_elements=primary_keys)

                    conn.execute(stmt)

                conn.commit()

            logger.info(f"Successfully upserted {len(df)} rows to {table_name}")

        except Exception as e:
            logger.error(f"Failed to upsert data to {table_name}: {str(e)}")
            raise

    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str, schema: Optional[str] = None, primary_keys: Optional[List[str]] = None) -> None:
        """
        Create a table based on DataFrame structure

        Args:
            df: DataFrame to base table structure on
            table_name: Name of the table to create
            schema: Schema name
            primary_keys: List of column names to set as primary keys
        """
        try:
            logger.info(f"Creating table {table_name} based on DataFrame structure")

            # Create table using pandas to_sql with sample data
            sample_df = df.head(0)  # Empty DataFrame with same structure
            sample_df.to_sql(name=table_name, con=self.engine, if_exists="replace", index=False, schema=schema)

            # Add primary key constraints if specified
            if primary_keys:
                with self.engine.connect() as conn:
                    pk_columns = ", ".join(primary_keys)
                    full_table_name = f"{schema}.{table_name}" if schema else table_name

                    alter_sql = f"""
                    ALTER TABLE {full_table_name}
                    ADD CONSTRAINT {table_name}_pkey PRIMARY KEY ({pk_columns})
                    """

                    conn.execute(alter_sql)
                    conn.commit()

            logger.info(f"Successfully created table {table_name}")

        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {str(e)}")
            raise

    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Execute raw SQL statement

        Args:
            sql: SQL statement to execute
            params: SQL parameters

        Returns:
            DataFrame if query returns results, None for DDL/DML statements
        """
        try:
            logger.info(f"Executing SQL: {sql[:100]}...")

            with self.engine.connect() as conn:
                result = conn.execute(sql, params or {})

                # Check if query returns data
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    logger.info(f"SQL executed successfully, returned {len(df)} rows")
                    return df
                else:
                    conn.commit()
                    logger.info("SQL executed successfully")
                    return None

        except Exception as e:
            logger.error(f"Failed to execute SQL: {str(e)}")
            raise

    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a table

        Args:
            table_name: Name of the table
            schema: Schema name

        Returns:
            Dictionary with table information
        """
        try:
            # Reflect table structure
            table = Table(table_name, self.metadata, autoload_with=self.engine, schema=schema)

            columns_info = []
            for column in table.columns:
                columns_info.append({"name": column.name, "type": str(column.type), "nullable": column.nullable, "primary_key": column.primary_key})

            # Get row count
            with self.engine.connect() as conn:
                from sqlalchemy import text

                # Safely quote table name to prevent SQL injection
                quoted_table = conn.dialect.identifier_preparer.quote(table_name)
                if schema:
                    quoted_schema = conn.dialect.identifier_preparer.quote(schema)
                    full_table_name = f"{quoted_schema}.{quoted_table}"
                else:
                    full_table_name = quoted_table
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {full_table_name}"))  # nosec B608
                row_count = count_result.scalar()

            return {"table_name": table_name, "schema": schema, "columns": columns_info, "row_count": row_count}

        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if hasattr(self, "engine"):
            self.engine.dispose()
