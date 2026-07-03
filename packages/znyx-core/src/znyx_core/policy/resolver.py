from typing import Dict, Any
from copy import deepcopy
from znyx_core.policy.loader import PolicyLoader
from znyx_core.utils.hash import compute_policy_hash
import logging

logger = logging.getLogger(__name__)


class PolicyResolver:
    """Resolves policy configuration with hierarchy: default -> tenant -> app -> agent -> env"""

    def __init__(self, loader: PolicyLoader):
        self.loader = loader

    def resolve(self, tenant_id: str, app_id: str, agent_id: str = "default", env: str = "prod") -> Dict[str, Any]:
        # Start with default policy
        resolved = deepcopy(self.loader.get_default_policy())

        # Apply tenant overrides
        tenant_policy = self.loader.get_tenant_policy(tenant_id)
        self._merge_policy(resolved, tenant_policy)

        # Apply app overrides
        app_policy = self.loader.get_app_policy(tenant_id, app_id)
        self._merge_policy(resolved, app_policy)

        # Apply agent overrides
        agent_policy = self.loader.get_agent_policy(tenant_id, app_id, agent_id)
        self._merge_policy(resolved, agent_policy)

        # Apply env overrides
        env_policy = self.loader.get_env_policy(tenant_id, app_id, agent_id, env)
        self._merge_policy(resolved, env_policy)

        # Compute policy version as hash of resolved policy
        policy_version = compute_policy_hash(resolved)
        resolved['policy_version'] = policy_version

        return resolved

    def _merge_policy(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if key in ['apps', 'agents', 'envs', 'tenants']:
                continue
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_policy(base[key], value)
            else:
                base[key] = deepcopy(value)
