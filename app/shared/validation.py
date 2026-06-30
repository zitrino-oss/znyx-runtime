"""Shared validation utilities for API routes."""
import re
from fastapi import HTTPException

_PATH_PARAM_RE = re.compile(r'^[a-zA-Z0-9_\-\.]{1,255}$')


def validate_path_params(**params: str) -> None:
    """Validate path parameters to prevent injection via path traversal or special chars."""
    for name, value in params.items():
        if not _PATH_PARAM_RE.match(value):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {name}: must be 1-255 alphanumeric/dash/underscore/dot characters"
            )
