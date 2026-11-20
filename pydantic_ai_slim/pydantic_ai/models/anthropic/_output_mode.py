"""Which structured output mode a request uses when its `output_type` doesn't pick one."""

from __future__ import annotations as _annotations

from typing import cast

from ..._json_schema import DEFINITION_KEYS
from ..._utils import is_str_dict

_SCHEMA_MAP_KEYS = ('properties', 'patternProperties', 'dependentSchemas', *DEFINITION_KEYS)
"""Keywords whose value maps names to schemas, so a name there is never itself a keyword."""


def has_open_object(node: object) -> bool:
    """Whether `node` has an object open to extra keys, like a `dict` field, which strict mode empties.

    Anthropic's strict schema transformation sets `additionalProperties` to `false` on every object, so the model
    can only return `{}` for one that had no fixed properties.
    """
    if is_str_dict(node):
        additional_properties = node.get('additionalProperties')
        if additional_properties is True or is_str_dict(additional_properties):
            return True
        for key, value in node.items():
            if key in _SCHEMA_MAP_KEYS and is_str_dict(value):
                if any(has_open_object(child) for child in value.values()):
                    return True
            elif has_open_object(value):
                return True
    elif isinstance(node, list):
        return any(has_open_object(child) for child in cast(list[object], node))
    return False
