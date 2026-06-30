import hashlib
import json
from typing import Any


def compute_policy_hash(policy_data: Any) -> str:
    """
    Compute a stable hash of policy data for versioning.

    Args:
        policy_data: Policy configuration data (dict, list, etc.)

    Returns:
        SHA256 hash string (first 16 characters)
    """
    # Convert to JSON string with sorted keys for stable hashing
    json_str = json.dumps(policy_data, sort_keys=True, separators=(',', ':'))

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))

    # Return first 16 characters of hex digest
    return hash_obj.hexdigest()[:16]
