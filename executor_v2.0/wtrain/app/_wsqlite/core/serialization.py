import json
from typing import Any, get_origin, get_args, Union
from datetime import datetime, date
from uuid import UUID

def extract_type(annotation):
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return args[0]
    return annotation

def is_json_type(annotation):
    t = extract_type(annotation)
    origin = get_origin(t)
    return t in (dict, list) or origin in (dict, list)

def serialize_value(val, annotation):
    if val is None:
        return None
    if is_json_type(annotation):
        return json.dumps(val)
    return val

def deserialize_value(val, annotation):
    if val is None:
        return None
    if is_json_type(annotation) and isinstance(val, str):
        try:
            return json.loads(val)
        except:
            return val
    return val
