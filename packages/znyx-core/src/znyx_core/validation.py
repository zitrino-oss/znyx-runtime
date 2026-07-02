"""Shared validation utilities for API routes."""
import re
from fastapi import HTTPException

_PATH_PARAM_RE = re.compile(r'^[a-zA-Z0-9_\-\.]{1,255}$')


def validate_path_params(**params: str) -> None:
    """Validate path parameters to prevent injection via path traversal or special chars."""
    for name, value in params.items():
        # The charset allows single dots (e.g. versioned env names) but a ".."
        # sequence is rejected outright so this can't be relied on to admit a
        # traversal token even if a caller ever uses these values as a path.
        if not _PATH_PARAM_RE.match(value) or ".." in value:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {name}: must be 1-255 alphanumeric/dash/underscore/dot characters (no '..')"
            )
