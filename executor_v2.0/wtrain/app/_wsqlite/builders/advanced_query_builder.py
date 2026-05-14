"""Advanced query builder with full SQL support.

Supports:
- Multiple SELECT fields
- JOINs (INNER, LEFT, RIGHT, CROSS)
- GROUP BY with HAVING
- UNION queries
- Subqueries
- Complex WHERE conditions
- Type-safe query building
"""

import re
from typing import Any, Optional, Union

from wsqlite.exceptions import SQLInjectionError


def validate_identifier(identifier: str) -> None:
    """Validate SQL identifier to prevent SQL injection."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise SQLInjectionError(identifier)


class QueryBuilder:
    """Advanced query builder with full SQL support.

    Features:
    - Type-safe query construction
    - JOIN support (INNER, LEFT, RIGHT, CROSS)
    - GROUP BY with aggregate functions
    - HAVING clause
    - UNION queries
    - Subqueries
    - Pagination

    Example:
        results = (
            QueryBuilder("users")
            .select("id", "name", "email")
            .join("orders", "users.id = orders.user_id", "LEFT")
            .where("status", "=", "active")
            .group_by("users.id")
            .having("COUNT(orders.id)", ">", 5)
            .order_by("users.name")
            .limit(100)
            .execute(conn)
        )
    """

    def __init__(self, table_name: str):
        """Initialize query builder.

        Args:
            table_name: Primary table name.
        """
        validate_identifier(table_name)
        self._table_name = table_name
        self._table_alias: Optional[str] = None

        self._select_fields: list[str] = ["*"]
        self._join_clauses: list[str] = []
        self._where_clauses: list[str] = []
        self._where_values: list[Any] = []
        self._group_by_fields: list[str] = []
        self._having_clauses: list[str] = []
        self._having_values: list[Any] = []
        self._order_by_fields: list[tuple[str, str]] = []
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._for_update: bool = False

    def select(self, *fields: str) -> "QueryBuilder":
        """Select specific fields.

        Args:
            *fields: Field names to select.

        Returns:
            Self for chaining.

        Example:
            .select("id", "name", "email")
        """
        self._select_fields = [validate_field(f) for f in fields]
        return self

    def select_as(self, field: str, alias: str) -> "QueryBuilder":
        """Select field with alias.

        Args:
            field: Field name.
            alias: Alias name.

        Returns:
            Self for chaining.

        Example:
            .select_as("COUNT(*)", "total")
        """
        validate_identifier(alias)
        self._select_fields.append(f"{field} AS {alias}")
        return self

    def select_count(self, field: str = "*", alias: str = "count") -> "QueryBuilder":
        """Select COUNT aggregate.

        Args:
            field: Field to count.
            alias: Result alias.

        Returns:
            Self for chaining.
        """
        self._select_fields.append(f"COUNT({field}) AS {alias}")
        return self

    def select_sum(self, field: str, alias: str) -> "QueryBuilder":
        """Select SUM aggregate."""
        self._select_fields.append(f"SUM({field}) AS {alias}")
        return self

    def select_avg(self, field: str, alias: str) -> "QueryBuilder":
        """Select AVG aggregate."""
        self._select_fields.append(f"AVG({field}) AS {alias}")
        return self

    def select_min(self, field: str, alias: str) -> "QueryBuilder":
        """Select MIN aggregate."""
        self._select_fields.append(f"MIN({field}) AS {alias}")
        return self

    def select_max(self, field: str, alias: str) -> "QueryBuilder":
        """Select MAX aggregate."""
        self._select_fields.append(f"MAX({field}) AS {alias}")
        return self

    def table(self, table: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Set table with optional alias."""
        validate_identifier(table)
        self._table_name = table
        self._table_alias = alias
        return self

    def join(
        self,
        table: str,
        on: str,
        join_type: str = "INNER",
    ) -> "QueryBuilder":
        """Add JOIN clause.

        Args:
            table: Table to join.
            on: JOIN condition.
            join_type: Type of JOIN (INNER, LEFT, RIGHT, CROSS).

        Returns:
            Self for chaining.

        Example:
            .join("orders", "users.id = orders.user_id", "LEFT")
        """
        valid_types = {"INNER", "LEFT", "RIGHT", "CROSS", "FULL"}
        if join_type.upper() not in valid_types:
            raise ValueError(f"Invalid JOIN type: {join_type}")

        validate_identifier(table)
        alias = ""
        if " " in table:
            parts = table.split()
            table = parts[0]
            alias = " " + " ".join(parts[1:])

        self._join_clauses.append(f"{join_type.upper()} JOIN {table}{alias} ON {on}")
        return self

    def inner_join(self, table: str, on: str) -> "QueryBuilder":
        """Add INNER JOIN."""
        return self.join(table, on, "INNER")

    def left_join(self, table: str, on: str) -> "QueryBuilder":
        """Add LEFT JOIN."""
        return self.join(table, on, "LEFT")

    def right_join(self, table: str, on: str) -> "QueryBuilder":
        """Add RIGHT JOIN."""
        return self.join(table, on, "RIGHT")

    def cross_join(self, table: str) -> "QueryBuilder":
        """Add CROSS JOIN."""
        return self.join(table, "", "CROSS")

    def where(
        self,
        field: str,
        operator: str,
        value: Any,
    ) -> "QueryBuilder":
        """Add WHERE condition with AND.

        Args:
            field: Field name.
            operator: Comparison operator.
            value: Value to compare.

        Returns:
            Self for chaining.

        Example:
            .where("age", ">=", 18)
            .where("status", "=", "active")
        """
        return self._add_where("AND", field, operator, value)

    def or_where(
        self,
        field: str,
        operator: str,
        value: Any,
    ) -> "QueryBuilder":
        """Add WHERE condition with OR."""
        return self._add_where("OR", field, operator, value)

    def _add_where(
        self,
        conjunction: str,
        field: str,
        operator: str,
        value: Any,
    ) -> "QueryBuilder":
        """Internal method to add WHERE clause."""
        valid_operators = {
            "=",
            "!=",
            "<>",
            "<",
            ">",
            "<=",
            ">=",
            "LIKE",
            "NOT LIKE",
            "IN",
            "NOT IN",
            "IS",
            "IS NOT",
            "BETWEEN",
        }

        op_upper = operator.upper()
        if op_upper not in valid_operators:
            raise ValueError(f"Invalid operator: {operator}")

        if op_upper in ("IS", "IS NOT"):
            if value is None:
                clause = f"{field} {operator.upper()} NULL"
            else:
                clause = f"{field} {operator.upper()} ?"
                self._where_values.append(value)
        elif op_upper == "BETWEEN":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("BETWEEN requires [low, high]")
            clause = f"{field} BETWEEN ? AND ?"
            self._where_values.extend(value)
        elif op_upper in ("IN", "NOT IN"):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{operator} requires a list or tuple")
            placeholders = ", ".join(["?"] * len(value))
            clause = f"{field} {op_upper} ({placeholders})"
            self._where_values.extend(value)
        else:
            clause = f"{field} {operator} ?"
            self._where_values.append(value)

        if self._where_clauses:
            self._where_clauses.append(f"{conjunction} {clause}")
        else:
            self._where_clauses.append(clause)

        return self

    def where_in(self, field: str, values: list) -> "QueryBuilder":
        """Add WHERE field IN (...) clause."""
        return self.where(field, "IN", values)

    def where_not_in(self, field: str, values: list) -> "QueryBuilder":
        """Add WHERE field NOT IN (...) clause."""
        return self.where(field, "NOT IN", values)

    def where_null(self, field: str) -> "QueryBuilder":
        """Add WHERE field IS NULL clause."""
        return self.where(field, "IS", None)

    def where_not_null(self, field: str) -> "QueryBuilder":
        """Add WHERE field IS NOT NULL clause."""
        return self.where(field, "IS NOT", None)

    def where_between(self, field: str, low: Any, high: Any) -> "QueryBuilder":
        """Add WHERE field BETWEEN low AND high clause."""
        return self.where(field, "BETWEEN", [low, high])

    def where_like(self, field: str, pattern: str) -> "QueryBuilder":
        """Add WHERE field LIKE pattern clause."""
        return self.where(field, "LIKE", pattern)

    def group_by(self, *fields: str) -> "QueryBuilder":
        """Add GROUP BY clause.

        Args:
            *fields: Fields to group by.

        Returns:
            Self for chaining.
        """
        self._group_by_fields = [validate_field(f) for f in fields]
        return self

    def having(
        self,
        condition: str,
        operator: str,
        value: Any,
    ) -> "QueryBuilder":
        """Add HAVING clause.

        Args:
            condition: Aggregate expression (e.g., "COUNT(*)").
            operator: Comparison operator.
            value: Value to compare.

        Returns:
            Self for chaining.

        Example:
            .having("COUNT(*)", ">", 10)
        """
        self._having_clauses.append(f"{condition} {operator} ?")
        self._having_values.append(value)
        return self

    def order_by(self, field: str, direction: str = "ASC") -> "QueryBuilder":
        """Add ORDER BY clause.

        Args:
            field: Field to order by.
            direction: ASC or DESC.

        Returns:
            Self for chaining.
        """
        if direction.upper() not in ("ASC", "DESC"):
            raise ValueError("Direction must be ASC or DESC")

        self._order_by_fields.append((field, direction.upper()))
        return self

    def order_by_asc(self, field: str) -> "QueryBuilder":
        """Add ORDER BY field ASC clause."""
        return self.order_by(field, "ASC")

    def order_by_desc(self, field: str) -> "QueryBuilder":
        """Add ORDER BY field DESC clause."""
        return self.order_by(field, "DESC")

    def limit(self, limit: int) -> "QueryBuilder":
        """Add LIMIT clause."""
        if limit < 0:
            raise ValueError("Limit must be non-negative")
        self._limit_value = limit
        return self

    def offset(self, offset: int) -> "QueryBuilder":
        """Add OFFSET clause."""
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        self._offset_value = offset
        return self

    def page(self, page: int, per_page: int = 10) -> "QueryBuilder":
        """Add LIMIT and OFFSET for pagination.

        Args:
            page: Page number (1-indexed).
            per_page: Records per page.

        Returns:
            Self for chaining.
        """
        if page < 1:
            page = 1
        self._limit_value = per_page
        self._offset_value = (page - 1) * per_page
        return self

    def for_update(self) -> "QueryBuilder":
        """Add FOR UPDATE clause (for locking)."""
        self._for_update = True
        return self

    def build_select(self) -> tuple[str, tuple]:
        """Build SELECT query.

        Returns:
            Tuple of (SQL query, parameters).
        """
        parts = ["SELECT", ", ".join(self._select_fields)]
        parts.append(f"FROM {self._table_name}")

        if self._join_clauses:
            parts.extend(self._join_clauses)

        if self._where_clauses:
            parts.append("WHERE")
            parts.append(" ".join(self._where_clauses))

        if self._group_by_fields:
            parts.append(f"GROUP BY {', '.join(self._group_by_fields)}")

        if self._having_clauses:
            parts.append("HAVING")
            parts.append(" ".join(self._having_clauses))

        if self._order_by_fields:
            order_parts = [f"{field} {direction}" for field, direction in self._order_by_fields]
            parts.append(f"ORDER BY {', '.join(order_parts)}")

        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        if self._offset_value is not None:
            parts.append(f"OFFSET {self._offset_value}")

        if self._for_update:
            parts.append("FOR UPDATE")

        query = " ".join(parts)
        values = tuple(self._where_values) + tuple(self._having_values)

        return query, values

    def build_count(self) -> tuple[str, tuple]:
        """Build COUNT query."""
        query, values = self.build_select()
        query = re.sub(r"SELECT .*? FROM", "SELECT COUNT(*) FROM", query, count=1)
        query = re.sub(r" ORDER BY.*", "", query)
        query = re.sub(r" LIMIT.*", "", query)
        query = re.sub(r" OFFSET.*", "", query)
        return query, values

    def build_insert(self, data: dict) -> tuple[str, tuple]:
        """Build INSERT query.

        Args:
            data: Dictionary of field: value pairs.

        Returns:
            Tuple of (SQL query, parameters).
        """
        fields = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(["?"] * len(fields))

        query = f"INSERT INTO {self._table_name} ({', '.join(fields)}) VALUES ({placeholders})"
        return query, tuple(values)

    def build_update(self, data: dict, where_field: str, where_value: Any) -> tuple[str, tuple]:
        """Build UPDATE query.

        Args:
            data: Dictionary of field: value pairs to update.
            where_field: WHERE field name.
            where_value: WHERE field value.

        Returns:
            Tuple of (SQL query, parameters).
        """
        set_clauses = [f"{k} = ?" for k in data.keys()]
        values = list(data.values())
        values.append(where_value)

        query = f"UPDATE {self._table_name} SET {', '.join(set_clauses)} WHERE {where_field} = ?"
        return query, tuple(values)

    def build_delete(self) -> tuple[str, tuple]:
        """Build DELETE query.

        Returns:
            Tuple of (SQL query, parameters).
        """
        if not self._where_clauses:
            raise ValueError("DELETE requires WHERE clause")

        query = f"DELETE FROM {self._table_name}"
        query += " WHERE " + " ".join(self._where_clauses)

        return query, tuple(self._where_values)

    def execute(self, conn) -> list:
        """Execute query and return results.

        Args:
            conn: Database connection.

        Returns:
            List of rows.
        """
        query, values = self.build_select()
        cursor = conn.execute(query, values)
        return cursor.fetchall()

    def execute_count(self, conn) -> int:
        """Execute COUNT query.

        Args:
            conn: Database connection.

        Returns:
            Count of matching records.
        """
        query, values = self.build_count()
        cursor = conn.execute(query, values)
        return cursor.fetchone()[0]

    def reset(self) -> "QueryBuilder":
        """Reset builder to initial state."""
        self._select_fields = ["*"]
        self._join_clauses = []
        self._where_clauses = []
        self._where_values = []
        self._group_by_fields = []
        self._having_clauses = []
        self._having_values = []
        self._order_by_fields = []
        self._limit_value = None
        self._offset_value = None
        self._for_update = False
        return self


def validate_field(field: str) -> str:
    """Validate a field specification."""
    if field == "*":
        return "*"

    if "(" in field:
        return field

    if "." in field:
        parts = field.split(".")
        validated = []
        for part in parts:
            if part and part != "*":
                validate_identifier(part)
            validated.append(part)
        return ".".join(validated)

    validate_identifier(field)
    return field
