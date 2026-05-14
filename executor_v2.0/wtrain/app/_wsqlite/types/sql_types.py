"""SQL type mapping from Pydantic to SQLite."""

import logging
from typing import Any, get_args, get_origin, Union
from datetime import datetime, date
from uuid import UUID
try:
    from decimal import Decimal
except ImportError:
    Decimal = float

logger = logging.getLogger(__name__)

try:
    from types import UnionType
except ImportError:
    UnionType = Union


def get_sql_type(field: Any) -> str:
    """Convert Pydantic field type to SQLite type.

    Maps Pydantic types to SQLite types:
    - int -> INTEGER
    - str -> TEXT
    - bool -> INTEGER (SQLite uses 0/1)
    - float/Decimal -> REAL
    - dict/list/datetime/UUID -> TEXT

    Supports constraints via field description:
    - "primary" -> PRIMARY KEY
    - "autoincrement" -> AUTOINCREMENT
    - "unique" -> UNIQUE (single column)
    - "not null" -> NOT NULL
    - "references:table.column" -> FOREIGN KEY
    """
    annotation = field.annotation
    
    # Handle Optional[T] and Union[T, None]
    origin = get_origin(annotation)
    if origin is Union or (UnionType is not Union and origin is UnionType):
        args = get_args(annotation)
        # Extract the non-None type from Union
        args = [arg for arg in args if arg is not type(None)]
        if args:
            annotation = args[0]
            origin = get_origin(annotation)

    # Handle generic types like dict[str, Any] or list[int]
    base_type = origin if origin is not None else annotation

    type_mapping = {
        int: "INTEGER", 
        str: "TEXT", 
        bool: "INTEGER",
        float: "REAL",
        Decimal: "REAL",
        datetime: "TEXT",
        date: "TEXT",
        UUID: "TEXT",
        dict: "TEXT",
        list: "TEXT",
    }
    
    # Fallback to TEXT if not in mapping
    sql_type = type_mapping.get(base_type, "TEXT")

    constraints = []
    description = (field.description or "").lower()
    if "primary" in description:
        constraints.append("PRIMARY KEY")
    if "autoincrement" in description:
        constraints.append("AUTOINCREMENT")
    
    # Only add UNIQUE if it's NOT a composite unique (unique:group)
    if "unique" in description and "unique:" not in description:
        constraints.append("UNIQUE")
        
    if "not null" in description:
        constraints.append("NOT NULL")

    return f"{sql_type} {' '.join(constraints)}".strip()
