"""
API data extractor
"""

from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class APIExtractor:
    """Extract data from REST APIs"""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 30):
        """
        Initialize API extractor

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            headers: Additional headers
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.API_BASE_URL
        self.api_key = api_key or settings.API_KEY
        self.timeout = timeout or settings.API_TIMEOUT

        self.headers = headers or {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def extract_endpoint(self, endpoint: str, params: Optional[Dict[str, Any]] = None, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], List[Any]]:
        """
        Extract data from a specific API endpoint

        Args:
            endpoint: API endpoint path
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
            data: Request body data

        Returns:
            Dictionary containing API response
        """
        try:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            logger.info(f"Making {method} request to: {url}")

            response = self.session.request(method=method, url=url, params=params, json=data, timeout=self.timeout)

            response.raise_for_status()

            json_data = response.json()
            logger.info(f"Successfully extracted data from API endpoint: {endpoint}")

            # Ensure we return a dictionary
            if isinstance(json_data, dict):
                return json_data
            else:
                # If it's not a dict, wrap it in one
                return {"data": json_data}

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to extract data from API {endpoint}: {str(e)}")
            raise

    def extract_paginated(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, page_param: str = "page", per_page_param: str = "per_page", per_page: int = 100, max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract data from paginated API endpoint

        Args:
            endpoint: API endpoint path
            params: Base query parameters
            page_param: Parameter name for page number
            per_page_param: Parameter name for items per page
            per_page: Number of items per page
            max_pages: Maximum number of pages to fetch

        Returns:
            List of all items from all pages
        """
        all_data = []
        page = 1
        params = params or {}

        while True:
            if max_pages and page > max_pages:
                break

            page_params = params.copy()
            page_params[page_param] = page
            page_params[per_page_param] = per_page

            try:
                response_data = self.extract_endpoint(endpoint, page_params)

                # Handle different response structures
                if isinstance(response_data, list):
                    items = response_data
                elif isinstance(response_data, dict):
                    if "data" in response_data:
                        items = response_data["data"]
                    elif "items" in response_data:
                        items = response_data["items"]
                    else:
                        items = [response_data]
                else:
                    items = [response_data]  # type: ignore[unreachable]

                if not items:
                    break

                all_data.extend(items)
                logger.info(f"Extracted page {page}, {len(items)} items")
                page += 1

            except Exception as e:
                logger.error(f"Error fetching page {page}: {str(e)}")
                break

        logger.info(f"Total extracted items: {len(all_data)}")
        return all_data

    def to_dataframe(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert API response data to pandas DataFrame

        Args:
            data: API response data

        Returns:
            DataFrame containing the data
        """
        if isinstance(data, dict):
            data = [data]

        return pd.DataFrame(data)

    def extract_to_dataframe(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        """
        Extract data from API endpoint and return as DataFrame

        Args:
            endpoint: API endpoint path
            **kwargs: Additional arguments for extract_endpoint

        Returns:
            DataFrame containing API data
        """
        data = self.extract_endpoint(endpoint, **kwargs)
        return self.to_dataframe(data)

    def close(self) -> None:
        """Close the session"""
        if hasattr(self, "session"):
            self.session.close()
