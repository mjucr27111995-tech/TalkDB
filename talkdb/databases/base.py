from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseDatabase(ABC):
    """Abstract base class for all database adapters."""

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Establish a connection to the database."""

    @abstractmethod
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame."""

    @abstractmethod
    def get_table_ddl(self, table_name: str) -> str:
        """Return the CREATE TABLE DDL for a given table."""

    @abstractmethod
    def get_all_ddls(self) -> dict[str, str]:
        """Return DDLs for every table in the database. Keys are table names."""

    @abstractmethod
    def get_dialect(self) -> str:
        """Return the SQL dialect name (e.g. 'sqlite', 'postgresql')."""

    @abstractmethod
    def close(self) -> None:
        """Close the database connection and release resources."""
