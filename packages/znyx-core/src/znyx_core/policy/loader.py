import logging
import os
import yaml
from typing import Dict, Any
from pathlib import Path

from znyx_core.policy.schema import validate_policy

logger = logging.getLogger(__name__)


class PolicyLoader:
    """Loads policy configuration from YAML file"""

    def __init__(self, policy_path: str = None):
        """
        Initialize policy loader.

        Args:
            policy_path: Path to policies.yaml file. If None, uses POLICY_PATH env var
                        or defaults to ./config/policies.yaml
        """
        if policy_path is None:
            policy_path = os.getenv('POLICY_PATH', './config/policies.yaml')

        self.policy_path = Path(policy_path)
        self._policies: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load policies from YAML file, validate the default + tenant policies, and
        serve the validated form.

        Validation used to run for its logging side effect only -- the returned
        ``PolicySchema`` was discarded and the raw YAML dict stayed in ``self._policies``,
        so a malformed key or a coercible-but-wrong-typed value was logged but still
        served as-is. Now the validated model's dict form is what gets served.

        Dumped with ``exclude_unset=True`` (not ``exclude_none``, and not a full dump):
        a field a scope never mentioned stays absent rather than materializing
        ``PolicySchema``'s default. This matters because ``PolicyResolver``'s deep-merge
        treats "key present in this scope" as "this scope decided this" -- a partial
        tenant/app/agent/env override that sets only e.g. ``threshold`` deliberately
        omits ``enabled`` to inherit it from a broader scope. Dumping with defaults
        filled in would make that omission explicit (e.g. ``enabled: false``, pydantic's
        default), silently re-deciding it and turning the detector off for that scope.
        ``exclude_unset=True`` reproduces the raw dict exactly when every field is valid,
        and only diverges where validation actually corrects something: an invalid key
        is genuinely dropped (not merely logged) and a coercible value is coerced.
        """
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")

        with open(self.policy_path, 'r') as f:
            self._policies = yaml.safe_load(f) or {}

        # Validate the default policy at load time and serve the validated form.
        default_policy = self._policies.get('default', {})
        if default_policy:
            validated = validate_policy(default_policy)
            self._policies['default'] = validated.model_dump(exclude_unset=True)
            logger.info("Default policy validated successfully")

        # Validate tenant-level policies and serve their validated form too.
        for tenant_id, tenant_policy in self._policies.get('tenants', {}).items():
            if isinstance(tenant_policy, dict):
                validated = validate_policy(tenant_policy)
                self._policies['tenants'][tenant_id] = validated.model_dump(exclude_unset=True)
                logger.debug("Tenant '%s' policy validated", tenant_id)

    def reload(self) -> None:
        """Reload policies from file"""
        self.load()

    def get_policies(self) -> Dict[str, Any]:
        """Get all loaded policies"""
        return self._policies

    def get_default_policy(self) -> Dict[str, Any]:
        """Get default policy configuration"""
        return self._policies.get('default', {})

    def get_tenant_policy(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant-specific policy"""
        tenants = self._policies.get('tenants', {})
        return tenants.get(tenant_id, {})

    def get_app_policy(self, tenant_id: str, app_id: str) -> Dict[str, Any]:
        """Get app-specific policy"""
        tenant_policy = self.get_tenant_policy(tenant_id)
        apps = tenant_policy.get('apps', {})
        return apps.get(app_id, {})

    def get_agent_policy(self, tenant_id: str, app_id: str, agent_id: str) -> Dict[str, Any]:
        """Get agent-specific policy"""
        app_policy = self.get_app_policy(tenant_id, app_id)
        agents = app_policy.get('agents', {})
        return agents.get(agent_id, {})

    def get_env_policy(self, tenant_id: str, app_id: str, agent_id: str, env: str) -> Dict[str, Any]:
        """Get environment-specific policy"""
        agent_policy = self.get_agent_policy(tenant_id, app_id, agent_id)
        envs = agent_policy.get('envs', {})
        return envs.get(env, {})
