"""Type validation and conversion utilities for wsqlite.

Provides:
- Type inference from Pydantic models
- Default value generation
- SQLite type mapping
- Field validation
"""

import json
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, get_origin, get_args, Union
from uuid import UUID


SQLITE_TYPE_MAP = {
    int: "INTEGER",
    str: "TEXT",
    float: "REAL",
    bool: "INTEGER",
    bytes: "BLOB",
    datetime: "TEXT",
    date: "TEXT",
    time: "TEXT",
    UUID: "TEXT",
    Decimal: "REAL",
}


PYTHON_TYPE_MAP = {
    "INTEGER": int,
    "REAL": float,
    "TEXT": str,
    "BLOB": bytes,
}


DEFAULT_VALUES: dict[type, Any] = {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    bytes: b"",
    list: [],
    dict: {},
    datetime: datetime(1970, 1, 1),
    date: date(1970, 1, 1),
    time: time(0, 0, 0),
    UUID: UUID("00000000-0000-0000-0000-000000000000"),
    Decimal: Decimal("0"),
}


def infer_sqlite_type(field_type: Any) -> str:
    """Infer SQLite column type from Python/Pydantic type.

    Args:
        field_type: Python type annotation.

    Returns:
        SQLite type string (INTEGER, TEXT, REAL, BLOB).

    Example:
        >>> infer_sqlite_type(int)
        'INTEGER'
        >>> infer_sqlite_type(str)
        'TEXT'
    """
    origin = get_origin(field_type)

    if field_type in SQLITE_TYPE_MAP:
        return SQLITE_TYPE_MAP[field_type]

    if origin is list or origin is set:
        return "TEXT"
    if origin is dict:
        return "TEXT"
    if origin is Union:
        args = get_args(field_type)
        for arg in args:
            if arg in SQLITE_TYPE_MAP:
                return SQLITE_TYPE_MAP[arg]

    for py_type, sql_type in SQLITE_TYPE_MAP.items():
        if field_type is py_type or (
            hasattr(field_type, "__origin__") and field_type.__origin__ is py_type
        ):
            return sql_type

    return "TEXT"


def get_default_value(field_type: Any) -> Any:
    """Get appropriate default value for a type.

    Args:
        field_type: Python type annotation.

    Returns:
        Default value for the type.

    Example:
        >>> get_default_value(int)
        0
        >>> get_default_value(str)
        ''
    """
    origin = get_origin(field_type)

    if field_type in DEFAULT_VALUES:
        return DEFAULT_VALUES[field_type]

    if origin is list or origin is set:
        return []
    if origin is dict:
        return {}

    if origin is Union:
        args = get_args(field_type)
        for arg in args:
            if arg in DEFAULT_VALUES:
                return DEFAULT_VALUES[arg]
            if arg is type(None):
                return None

    return None


def validate_value(field_type: Any, value: Any) -> Any:
    """Validate and convert a value to the correct type.

    Args:
        field_type: Expected Python type.
        value: Value to validate.

    Returns:
        Validated and converted value.

    Raises:
        ValueError: If value cannot be converted.
    """
    if value is None:
        return get_default_value(field_type)

    origin = get_origin(field_type)

    if field_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    if field_type is int:
        if isinstance(value, int):
            return value
        return int(value)

    if field_type is float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(value)

    if field_type is str:
        return str(value)

    if field_type is bytes:
        if isinstance(value, bytes):
            return value
        return bytes(value)

    if field_type is datetime or origin is datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(str(value))

    if field_type is date or origin is date:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        return date.fromisoformat(str(value))

    if field_type is time or origin is time:
        if isinstance(value, time):
            return value
        if isinstance(value, str):
            return time.fromisoformat(value)
        return time.fromisoformat(str(value))

    if field_type is UUID or origin is UUID:
        if isinstance(value, UUID):
            return value
        return UUID(str(value))

    if origin is list or origin is set:
        if isinstance(value, (list, set)):
            return list(value)
        if isinstance(value, str):
            return json.loads(value)
        return [value]

    if origin is dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return {}

    if field_type is dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return {}

    return value


def serialize_value(value: Any, field_type: Any) -> Any:
    """Serialize a value for SQLite storage.

    Args:
        value: Value to serialize.
        field_type: Target Python type.

    Returns:
        Serializable value for SQLite.
    """
    if value is None:
        return None

    if field_type is bool:
        return 1 if value else 0

    if field_type in (datetime, date, time):
        if isinstance(value, str):
            return value
        return value.isoformat()

    if field_type is UUID:
        return str(value)

    if field_type in (list, set, dict) or get_origin(field_type) in (list, set, dict):
        return json.dumps(value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (list, dict)):
        return json.dumps(value)

    return value


def deserialize_value(value: Any, field_type: Any) -> Any:
    """Deserialize a value from SQLite storage.

    Args:
        value: Raw SQLite value.
        field_type: Target Python type.

    Returns:
        Deserialized value.
    """
    if value is None:
        return get_default_value(field_type)

    return validate_value(field_type, value)


class FieldValidator:
    """Validator for database fields.

    Provides type checking and conversion for model fields.

    Example:
        validator = FieldValidator(model)

        # Validate and convert a record
        cleaned = validator.clean(record)

        # Check field type
        validator.is_valid_type("field_name", value)
    """

    def __init__(self, model: type):
        """Initialize validator with Pydantic model.

        Args:
            model: Pydantic BaseModel class.
        """
        self.model = model
        self._fields = {}

        if hasattr(model, "model_fields"):
            for name, field_info in model.model_fields.items():
                self._fields[name] = {
                    "type": field_info.annotation,
                    "required": field_info.is_required(),
                    "default": field_info.default,
                }

    def clean(self, data: dict) -> dict:
        """Clean and validate a data dictionary.

        Args:
            data: Raw data dictionary.

        Returns:
            Cleaned and validated dictionary.
        """
        cleaned = {}

        for name, value in data.items():
            if name in self._fields:
                field_info = self._fields[name]
                cleaned[name] = serialize_value(value, field_info["type"])
            else:
                cleaned[name] = value

        return cleaned

    def is_valid_type(self, field_name: str, value: Any) -> bool:
        """Check if value matches field type.

        Args:
            field_name: Name of field.
            value: Value to check.

        Returns:
            True if valid, False otherwise.
        """
        if field_name not in self._fields:
            return True

        field_type = self._fields[field_name]["type"]

        try:
            deserialize_value(value, field_type)
            return True
        except (ValueError, TypeError):
            return False

    def get_sql_type(self, field_name: str) -> str:
        """Get SQLite type for a field.

        Args:
            field_name: Name of field.

        Returns:
            SQLite type string.
        """
        if field_name not in self._fields:
            return "TEXT"

        return infer_sqlite_type(self._fields[field_name]["type"])
