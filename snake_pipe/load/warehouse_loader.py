"""
Data warehouse loader
"""

from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WarehouseLoader:
    """Load data to data warehouses (Snowflake, BigQuery, Redshift, etc.)"""

    def __init__(self, warehouse_type: str, connection_params: Dict[str, Any]) -> None:
        """
        Initialize warehouse loader

        Args:
            warehouse_type: Type of warehouse ('snowflake', 'bigquery', 'redshift')
            connection_params: Connection parameters specific to the warehouse
        """
        self.warehouse_type = warehouse_type.lower()
        self.connection_params = connection_params
        logger.info(f"Initialized WarehouseLoader for {warehouse_type}")

        # Initialize connection based on warehouse type
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize connection to the specific warehouse"""
        if self.warehouse_type == "snowflake":
            self._init_snowflake()
        elif self.warehouse_type == "bigquery":
            self._init_bigquery()
        elif self.warehouse_type == "redshift":
            self._init_redshift()
        else:
            raise ValueError(f"Unsupported warehouse type: {self.warehouse_type}")

    def _init_snowflake(self) -> None:
        """Initialize Snowflake connection"""
        try:
            import snowflake.connector
            from snowflake.connector.pandas_tools import write_pandas

            self.connection = snowflake.connector.connect(**self.connection_params)
            self.write_pandas = write_pandas
            logger.info("Snowflake connection initialized")

        except ImportError:
            raise ImportError("snowflake-connector-python is required for Snowflake support")
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {str(e)}")
            raise

    def _init_bigquery(self) -> None:
        """Initialize BigQuery connection"""
        try:
            from google.cloud import bigquery

            self.client = bigquery.Client(**self.connection_params)
            logger.info("BigQuery connection initialized")

        except ImportError:
            raise ImportError("google-cloud-bigquery is required for BigQuery support")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery connection: {str(e)}")
            raise

    def _init_redshift(self) -> None:
        """Initialize Redshift connection"""
        try:
            from sqlalchemy import create_engine

            # Build connection string for Redshift
            conn_str = (
                f"postgresql://{self.connection_params['user']}:"
                f"{self.connection_params['password']}@"
                f"{self.connection_params['host']}:"
                f"{self.connection_params['port']}/"
                f"{self.connection_params['database']}"
            )

            self.engine = create_engine(conn_str)
            logger.info("Redshift connection initialized")

        except ImportError:
            raise ImportError("psycopg2-binary is required for Redshift support")
        except Exception as e:
            logger.error(f"Failed to initialize Redshift connection: {str(e)}")
            raise

    def load_dataframe(self, df: pd.DataFrame, table_name: str, database: Optional[str] = None, schema: Optional[str] = None, if_exists: str = "replace", **kwargs: Any) -> None:
        """
        Load DataFrame to warehouse table

        Args:
            df: DataFrame to load
            table_name: Name of the target table
            database: Database name (warehouse-specific)
            schema: Schema name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            **kwargs: Additional warehouse-specific parameters
        """
        try:
            logger.info(f"Loading {len(df)} rows to {self.warehouse_type} table {table_name}")

            if self.warehouse_type == "snowflake":
                self._load_to_snowflake(df, table_name, database, schema, if_exists, **kwargs)
            elif self.warehouse_type == "bigquery":
                self._load_to_bigquery(df, table_name, database, schema, if_exists, **kwargs)
            elif self.warehouse_type == "redshift":
                self._load_to_redshift(df, table_name, schema, if_exists, **kwargs)

            logger.info(f"Successfully loaded {len(df)} rows to {table_name}")

        except Exception as e:
            logger.error(f"Failed to load data to {table_name}: {str(e)}")
            raise

    def _load_to_snowflake(self, df: pd.DataFrame, table_name: str, database: Optional[str], schema: Optional[str], if_exists: str, **kwargs: Any) -> None:
        """Load data to Snowflake"""
        cursor = self.connection.cursor()

        try:
            # Set database and schema if provided
            if database:
                cursor.execute(f"USE DATABASE {database}")
            if schema:
                cursor.execute(f"USE SCHEMA {schema}")

            # Create table if it doesn't exist and if_exists is 'replace'
            if if_exists == "replace":
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Use write_pandas to load data
            success, _, _, _ = self.write_pandas(
                conn=self.connection, df=df, table_name=table_name, database=database, schema=schema, auto_create_table=True, overwrite=(if_exists == "replace"), **kwargs
            )

            if not success:
                raise Exception("Failed to write data to Snowflake")

        finally:
            cursor.close()

    def _load_to_bigquery(self, df: pd.DataFrame, table_name: str, dataset: Optional[str], schema: Optional[str], if_exists: str, **kwargs: Any) -> None:
        """Load data to BigQuery"""
        # BigQuery uses dataset instead of database
        dataset_id = dataset or self.connection_params.get("dataset_id")
        if not dataset_id:
            raise ValueError("Dataset ID is required for BigQuery")

        table_ref = self.client.dataset(dataset_id).table(table_name)

        job_config = self.client.LoadJobConfig()

        # Set write disposition based on if_exists
        if if_exists == "replace":
            job_config.write_disposition = self.client.WriteDisposition.WRITE_TRUNCATE
        elif if_exists == "append":
            job_config.write_disposition = self.client.WriteDisposition.WRITE_APPEND
        else:  # fail
            job_config.write_disposition = self.client.WriteDisposition.WRITE_EMPTY

        job_config.autodetect = True

        # Load data
        job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for job to complete

    def _load_to_redshift(self, df: pd.DataFrame, table_name: str, schema: Optional[str], if_exists: str, **kwargs: Any) -> None:
        """Load data to Redshift"""
        df.to_sql(name=table_name, con=self.engine, schema=schema, if_exists=if_exists, index=False, method="multi", **kwargs)  # Use multi-row insert for better performance

    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Execute SQL statement

        Args:
            sql: SQL statement to execute
            params: SQL parameters

        Returns:
            DataFrame if query returns results, None for DDL/DML statements
        """
        try:
            logger.info(f"Executing SQL on {self.warehouse_type}: {sql[:100]}...")

            if self.warehouse_type == "snowflake":
                return self._execute_snowflake_sql(sql, params)
            elif self.warehouse_type == "bigquery":
                return self._execute_bigquery_sql(sql, params)
            elif self.warehouse_type == "redshift":
                return self._execute_redshift_sql(sql, params)
            else:
                raise ValueError(f"Unsupported warehouse type: {self.warehouse_type}")

        except Exception as e:
            logger.error(f"Failed to execute SQL: {str(e)}")
            raise

    def _execute_snowflake_sql(self, sql: str, params: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Execute SQL on Snowflake"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params)

            if cursor.description:  # Query returns data
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                return pd.DataFrame(data, columns=columns)
            return None
        finally:
            cursor.close()

    def _execute_bigquery_sql(self, sql: str, params: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Execute SQL on BigQuery"""
        job_config = self.client.QueryJobConfig()
        if params:
            # Convert parameters to BigQuery format
            query_parameters = []
            for key, value in params.items():
                query_parameters.append(self.client.ScalarQueryParameter(key, "STRING", str(value)))
            job_config.query_parameters = query_parameters

        query_job = self.client.query(sql, job_config=job_config)
        results = query_job.result()

        return results.to_dataframe() if results.total_rows > 0 else None

    def _execute_redshift_sql(self, sql: str, params: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Execute SQL on Redshift"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, params or {})

                if result.returns_rows:
                    return pd.DataFrame(result.fetchall(), columns=result.keys())
                return None
        except Exception as e:
            raise e

    def get_table_info(self, table_name: str, database: Optional[str] = None, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a warehouse table

        Args:
            table_name: Name of the table
            database: Database/dataset name
            schema: Schema name

        Returns:
            Dictionary with table information
        """
        if self.warehouse_type == "snowflake":
            sql = f"DESCRIBE TABLE {table_name}"
        elif self.warehouse_type == "bigquery":
            dataset = database or self.connection_params.get("dataset_id")
            table_ref = self.client.dataset(dataset).table(table_name)
            table = self.client.get_table(table_ref)

            return {
                "table_name": table_name,
                "dataset": dataset,
                "rows": table.num_rows,
                "columns": len(table.schema),
                "schema": [{"name": field.name, "type": field.field_type} for field in table.schema],
            }
        elif self.warehouse_type == "redshift":
            from sqlalchemy import text

            # Use parameterized query for table name safety
            sql = text(
                """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = :table_name
            AND table_schema = :schema
            """
            )

            params = {"table_name": table_name, "schema": schema or "public"}

        if self.warehouse_type != "bigquery":
            result_df = self.execute_sql(sql, params if self.warehouse_type == "redshift" else None)
            return {"table_name": table_name, "database": database, "schema": schema, "columns": result_df.to_dict("records") if result_df is not None else []}

        return {}

    def close(self) -> None:
        """Close warehouse connection"""
        try:
            if hasattr(self, "connection"):
                self.connection.close()
            elif hasattr(self, "client"):
                self.client.close()
            elif hasattr(self, "engine"):
                self.engine.dispose()

            logger.info(f"Closed {self.warehouse_type} connection")
        except Exception as e:
            logger.warning(f"Error closing connection: {str(e)}")
