"""Table synchronization with Pydantic models."""

import re
from typing import Optional

from wsqlite.core.connection import get_async_connection, get_connection
from wsqlite.types.sql_types import get_sql_type


def validate_identifier(identifier: str) -> None:
    """Validate SQL identifier to prevent SQL injection."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        from wsqlite.exceptions import SQLInjectionError

        raise SQLInjectionError(f"Invalid identifier: {identifier}")


class TableSync:
    """Handles table synchronization between Pydantic models and SQLite (sync)."""

    def __init__(self, model, db_path: str, table_name: Optional[str] = None):
        """Initialize table sync.

        Args:
            model: Pydantic BaseModel class.
            db_path: Path to SQLite database file.
            table_name: Optional custom table name.
        """
        self.model = model
        self.db_path = db_path
        self.table_name = table_name or model.__name__.lower()

    def create_if_not_exists(self):
        """Create the table if it doesn't exist, handling FTS5 virtual tables."""
        config = getattr(self.model, "wsqlite_config", None)
        use_fts = getattr(config, "use_fts5", False)

        if use_fts:
            # Handle FTS5 table creation
            fts_columns = [
                field_name
                for field_name, field in self.model.model_fields.items()
                if get_sql_type(field) == "TEXT"
            ]
            if not fts_columns:
                raise TableSyncError("FTS5 table requires at least one TEXT field.")
            
            columns_clause = ", ".join(fts_columns)
            query = f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING fts5({columns_clause})"
        else:
            # Standard table creation
            column_defs = []
            composite_uniques = {}
            foreign_keys = []

            for field_name, field in self.model.model_fields.items():
                col_type = get_sql_type(field)
                column_defs.append(f"{field_name} {col_type}")

                description = (field.description or "").lower()
                if "unique:" in description:
                    match = re.search(r"unique:([a-zA-Z0-9_]+)", description)
                    if match:
                        group = match.group(1)
                        composite_uniques.setdefault(group, []).append(field_name)
                if "references:" in description:
                    match = re.search(r"references:([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", description)
                    if match:
                        foreign_keys.append((field_name, match.group(1), match.group(2)))

            for fields in composite_uniques.values():
                column_defs.append(f"UNIQUE({', '.join(fields)})")
            for local_col, ref_table, ref_col in foreign_keys:
                column_defs.append(f"FOREIGN KEY({local_col}) REFERENCES {ref_table}({ref_col})")

            fields_clause = ", ".join(column_defs)
            query = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({fields_clause})"

        with get_connection(self.db_path) as conn:
            conn.execute(query)
            conn.commit()

        # Auto-create indexes for non-FTS tables
        if not use_fts:
            for field_name, field in self.model.model_fields.items():
                description = (field.description or "").lower()
                if "index" in description:
                    unique = "unique" in description and "unique:" not in description
                    self.create_index([field_name], unique=unique)

    def sync_with_model(self):
        """Sync the table with the Pydantic model, adding new columns if necessary."""
        # FTS5 tables cannot be altered
        config = getattr(self.model, "wsqlite_config", None)
        if getattr(config, "use_fts5", False):
            return

        query = f"PRAGMA table_info({self.table_name})"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query)
            existing_columns = {row[1] for row in cursor.fetchall()}

        model_fields = set(self.model.model_fields.keys())
        new_fields = model_fields - existing_columns

        if new_fields:
            with get_connection(self.db_path) as conn:
                for field in new_fields:
                    field_type = get_sql_type(self.model.model_fields[field])
                    alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {field} {field_type}"
                    conn.execute(alter_query)
                conn.commit()

    def table_exists(self) -> bool:
        """Check if the table exists in the database."""
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (self.table_name,))
            return cursor.fetchone() is not None

    def drop_table(self):
        """Drop the table from the database."""
        query = f"DROP TABLE IF EXISTS {self.table_name}"
        with get_connection(self.db_path) as conn:
            conn.execute(query)
            conn.commit()

    def get_columns(self) -> list[str]:
        """Get list of column names in the table."""
        query = f"PRAGMA table_info({self.table_name})"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query)
            return [row[1] for row in cursor.fetchall()]

    def create_index(
        self, columns: list[str], index_name: Optional[str] = None, unique: bool = False
    ):
        """Create an index on the specified columns."""
        if index_name is None:
            index_name = f"idx_{self.table_name}_{'_'.join(columns)}"

        columns_str = ", ".join(columns)
        unique_str = "UNIQUE " if unique else ""
        query = f"CREATE {unique_str}INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({columns_str})"

        with get_connection(self.db_path) as conn:
            conn.execute(query)
            conn.commit()

    def drop_index(self, index_name: str):
        """Drop an index from the table."""
        query = f"DROP INDEX IF EXISTS {index_name}"
        with get_connection(self.db_path) as conn:
            conn.execute(query)
            conn.commit()

    def get_indexes(self) -> list[dict]:
        """Get list of indexes on the table."""
        query = f"PRAGMA index_list({self.table_name})"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query)
            indexes = []
            for row in cursor.fetchall():
                idx_name = row[1]
                idx_info = f"PRAGMA index_info({idx_name})"
                idx_cursor = conn.execute(idx_info)
                col_names = [col[2] for col in idx_cursor.fetchall()]
                indexes.append(
                    {
                        "name": idx_name,
                        "unique": bool(row[2]),
                        "columns": col_names,
                    }
                )
            return indexes


class AsyncTableSync:
    """Handles table synchronization between Pydantic models and SQLite (async)."""

    def __init__(self, model, db_path: str, table_name: Optional[str] = None):
        """Initialize async table sync."""
        self.model = model
        self.db_path = db_path
        self.table_name = table_name or model.__name__.lower()

    async def create_if_not_exists_async(self):
        """Create the table if it doesn't exist (async)."""
        column_defs = []
        composite_uniques = {}
        foreign_keys = []

        for field_name, field in self.model.model_fields.items():
            col_type = get_sql_type(field)
            column_defs.append(f"{field_name} {col_type}")

            description = (field.description or "").lower()

            if "unique:" in description:
                match = re.search(r"unique:([a-zA-Z0-9_]+)", description)
                if match:
                    group = match.group(1)
                    if group not in composite_uniques:
                        composite_uniques[group] = []
                    composite_uniques[group].append(field_name)

            if "references:" in description:
                match = re.search(r"references:([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", description)
                if match:
                    ref_table = match.group(1)
                    ref_col = match.group(2)
                    foreign_keys.append((field_name, ref_table, ref_col))

        for fields in composite_uniques.values():
            fields_str = ", ".join(fields)
            column_defs.append(f"UNIQUE({fields_str})")

        for local_col, ref_table, ref_col in foreign_keys:
            column_defs.append(f"FOREIGN KEY({local_col}) REFERENCES {ref_table}({ref_col})")

        fields_clause = ", ".join(column_defs)
        query = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({fields_clause})"
        
        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query)
            await conn.commit()
        finally:
            await conn.close()

        # Auto-create indexes
        for field_name, field in self.model.model_fields.items():
            description = (field.description or "").lower()
            if "index" in description:
                unique = "unique" in description and "unique:" not in description
                await self.create_index_async([field_name], unique=unique)

    async def sync_with_model_async(self):
        """Sync the table with the Pydantic model, adding new columns if necessary (async)."""
        query = f"PRAGMA table_info({self.table_name})"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query)
            rows = await cursor.fetchall()
            existing_columns = {row[1] for row in rows}
        finally:
            await conn.close()

        model_fields = set(self.model.model_fields.keys())
        new_fields = model_fields - existing_columns

        if new_fields:
            conn = await get_async_connection(self.db_path)
            try:
                for field in new_fields:
                    field_type = get_sql_type(self.model.model_fields[field])
                    alter_query = f"ALTER TABLE {self.table_name} ADD COLUMN {field} {field_type}"
                    await conn.execute(alter_query)
                await conn.commit()
            finally:
                await conn.close()

    async def table_exists_async(self) -> bool:
        """Check if the table exists in the database (async)."""
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query, (self.table_name,))
            result = await cursor.fetchone()
            return result is not None
        finally:
            await conn.close()

    async def drop_table_async(self):
        """Drop the table from the database (async)."""
        query = f"DROP TABLE IF EXISTS {self.table_name}"
        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query)
            await conn.commit()
        finally:
            await conn.close()

    async def get_columns_async(self) -> list[str]:
        """Get list of column names in the table (async)."""
        query = f"PRAGMA table_info({self.table_name})"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query)
            rows = await cursor.fetchall()
            return [row[1] for row in rows]
        finally:
            await conn.close()

    async def create_index_async(
        self, columns: list[str], index_name: Optional[str] = None, unique: bool = False
    ):
        """Create an index on the specified columns (async)."""
        if index_name is None:
            index_name = f"idx_{self.table_name}_{'_'.join(columns)}"

        columns_str = ", ".join(columns)
        unique_str = "UNIQUE " if unique else ""
        query = f"CREATE {unique_str}INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({columns_str})"

        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query)
            await conn.commit()
        finally:
            await conn.close()

    async def drop_index_async(self, index_name: str):
        """Drop an index from the table (async)."""
        query = f"DROP INDEX IF EXISTS {index_name}"
        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query)
            await conn.commit()
        finally:
            await conn.close()

    async def get_indexes_async(self) -> list[dict]:
        """Get list of indexes on the table (async)."""
        query = f"PRAGMA index_list({self.table_name})"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query)
            indexes = []
            for row in await cursor.fetchall():
                idx_name = row[1]
                idx_info = f"PRAGMA index_info({idx_name})"
                idx_cursor = await conn.execute(idx_info)
                col_names = [col[2] for col in await idx_cursor.fetchall()]
                indexes.append(
                    {
                        "name": idx_name,
                        "unique": bool(row[2]),
                        "columns": col_names,
                    }
                )
            return indexes
        finally:
            await conn.close()
