import logging
import os
import yaml
from typing import Dict, Any
from pathlib import Path

from app.shared.policy.schema import validate_policy

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
        """Load policies from YAML file and validate the default policy."""
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")

        with open(self.policy_path, 'r') as f:
            self._policies = yaml.safe_load(f) or {}

        # Validate the default policy at load time
        default_policy = self._policies.get('default', {})
        if default_policy:
            validated = validate_policy(default_policy)
            logger.info("Default policy validated successfully")

        # Validate tenant-level policies
        for tenant_id, tenant_policy in self._policies.get('tenants', {}).items():
            if isinstance(tenant_policy, dict):
                validate_policy(tenant_policy)
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
