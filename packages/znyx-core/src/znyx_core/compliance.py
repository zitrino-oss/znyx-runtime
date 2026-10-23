"""Compliance crosswalks (AI gateway Phase 5, part 4): what ZNYX can evidence, per framework.

Three frameworks, each a list of requirements; a requirement names the checks that evidence it
(``CHECKS``). The control plane evaluates the checks for an org (``True``: the evidence is there,
``False``: it is not, ``None``: it could not be determined) and ``assess`` turns them into a
status per requirement:

- ``supported``: every check passed;
- ``partial``: some passed;
- ``not_yet``: none passed;
- ``organisational``: a process or document ZNYX cannot evidence (a risk assessment, training,
  a notice to users). The org evidences it elsewhere.

The wording states what the evidence shows; it never claims that an org complies. Requirement
texts are short paraphrases for orientation, not the frameworks' text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

STATUSES = ("supported", "partial", "not_yet", "organisational")

DISCLAIMER = ("This shows what ZNYX records evidence for. It is not a statement of compliance: the "
              "frameworks also need processes and documents outside ZNYX, and a qualified assessor "
              "decides compliance.")

# check -> what passing it means. The control plane evaluates each for an org.
CHECKS: Dict[str, str] = {
    "policy_published": "Every environment of every project serves a published guardrail policy.",
    "approval_rules": "A published policy requires a person's approval for chosen tools or destructive MCP tools.",
    "kill_switch_ready": "A runtime polled the controls in the last 24 hours, so a stop applies within one poll.",
    "audit_chain_verified": "The control plane's audit chain and every runtime audit chain verify, with no gaps "
                            "or mismatches.",
    "runtime_audit_shipping": "Runtime audit records (egress or governance) arrived in the last 7 days.",
    "usage_observability": "Usage records of gateway calls arrived in the last 7 days.",
    "log_retention": "Usage and audit records are kept for at least six months (183 days).",
    "pii_protection_production": "PII protection (redact or block) is on in every production environment's policy.",
    "secrets_protection": "Secrets detection is on in every published policy.",
    "prompt_injection_protection": "Prompt-injection or jailbreak detection is on in every published policy.",
    "owasp_coverage": "The OWASP Top 10 for LLM applications coverage score is at least 80.",
    "testing_evaluations": "A benchmark or model evaluation ran in the last 90 days.",
    "model_monitoring": "Model drift checks or evaluation history exist for the models in use.",
    "egress_allowlist": "Outbound AI calls are limited to listed providers and hosts (an egress allowlist).",
    "inventory_reviewed": "Every AI inventory item (models, providers, tools, MCP servers, agents) has been "
                          "reviewed, and none waits longer than 30 days.",
    "shadow_ai_discovery": "Shadow-AI discovery ran in the last 30 days.",
    "mcp_change_review": "No MCP tool definition change has waited for review longer than 7 days.",
    "agent_identity": "An identity provider is configured and agents act for people with delegated tokens.",
    "prompt_versioning": "Prompt versions (fingerprints) are recorded for gateway calls.",
    "budgets_set": "Spend budgets are set for the org or its projects.",
}


@dataclass(frozen=True)
class Requirement:
    id: str                         # stable within its framework, e.g. "art-12"
    ref: str                        # how the framework names it, e.g. "Article 12"
    title: str
    summary: str                    # a short paraphrase
    checks: Tuple[str, ...] = ()    # empty: organisational
    note: Optional[str] = None      # what ZNYX's evidence covers and what it leaves to the org


@dataclass(frozen=True)
class Framework:
    id: str
    title: str
    version: str
    source: str
    requirements: Tuple[Requirement, ...] = field(default_factory=tuple)


def _r(id: str, ref: str, title: str, summary: str, checks: Tuple[str, ...] = (), note: Optional[str] = None):
    return Requirement(id=id, ref=ref, title=title, summary=summary, checks=checks, note=note)


EU_AI_ACT = Framework(
    id="eu_ai_act", title="EU AI Act", version="Regulation (EU) 2024/1689",
    source="https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
    requirements=(
        _r("art-4", "Article 4", "AI literacy",
           "Staff who operate or use AI systems have a sufficient level of AI literacy."),
        _r("art-9", "Article 9", "Risk management system",
           "A risk management system runs across the lifecycle of a high-risk AI system, with testing.",
           ("owasp_coverage", "testing_evaluations", "inventory_reviewed"),
           "ZNYX evidences testing, coverage of known LLM risks and an inventory; the risk management "
           "process itself is the org's."),
        _r("art-10", "Article 10", "Data and data governance",
           "Training, validation and testing data meet quality criteria.",
           note="ZNYX does not train models or hold training data."),
        _r("art-11", "Article 11", "Technical documentation",
           "Technical documentation (Annex IV) is drawn up and kept up to date.",
           ("inventory_reviewed", "policy_published"),
           "The evidence pack documents the AI inventory and the policies in force; the rest of Annex IV is "
           "the provider's."),
        _r("art-12", "Article 12", "Record-keeping",
           "The system automatically records events (logs) over its lifetime, for traceability.",
           ("audit_chain_verified", "runtime_audit_shipping", "usage_observability")),
        _r("art-13", "Article 13", "Transparency and information to deployers",
           "The system comes with instructions for use that deployers can understand."),
        _r("art-14", "Article 14", "Human oversight",
           "People can oversee the system, decide not to use an output, and interrupt it with a stop.",
           ("approval_rules", "kill_switch_ready"),
           "Approvals hold chosen actions for a person; the kill switch stops keys, apps, agents, models, "
           "tools, MCP servers or people. Assigning competent overseers is the org's."),
        _r("art-15", "Article 15", "Accuracy, robustness and cybersecurity",
           "The system resists errors and attempts to alter its use or outputs (including adversarial inputs).",
           ("prompt_injection_protection", "owasp_coverage", "testing_evaluations")),
        _r("art-17", "Article 17", "Quality management system",
           "Providers of high-risk systems run a documented quality management system."),
        _r("art-19", "Article 19", "Automatically generated logs",
           "Providers keep the logs their systems generate for at least six months.",
           ("audit_chain_verified", "log_retention")),
        _r("art-26", "Article 26", "Obligations of deployers",
           "Deployers use systems as instructed, assign human oversight, monitor operation and keep logs for "
           "at least six months.",
           ("approval_rules", "usage_observability", "log_retention"),
           "Using the system as instructed and informing affected people remain the org's."),
        _r("art-27", "Article 27", "Fundamental rights impact assessment",
           "Certain deployers assess the impact on fundamental rights before first use."),
        _r("art-50", "Article 50", "Transparency obligations for certain AI systems",
           "People are told they are interacting with an AI system; synthetic content is marked."),
        _r("art-72", "Article 72", "Post-market monitoring",
           "Providers collect and analyse data on the system's performance throughout its lifetime.",
           ("usage_observability", "model_monitoring")),
        _r("art-73", "Article 73", "Reporting of serious incidents",
           "Serious incidents are reported to the market surveillance authorities.",
           note="Audit records and traces support an incident investigation; reporting is the org's."),
    ),
)

NIST_AI_RMF = Framework(
    id="nist_ai_rmf", title="NIST AI Risk Management Framework", version="AI RMF 1.0 (NIST AI 100-1)",
    source="https://doi.org/10.6028/NIST.AI.100-1",
    requirements=(
        _r("gv-1.1", "GOVERN 1.1", "Legal and regulatory requirements",
           "Legal and regulatory requirements involving AI are understood, managed and documented."),
        _r("gv-1.6", "GOVERN 1.6", "AI system inventory",
           "Mechanisms are in place to inventory AI systems.",
           ("inventory_reviewed", "shadow_ai_discovery")),
        _r("gv-2.1", "GOVERN 2.1", "Roles and responsibilities",
           "Roles, responsibilities and lines of communication for AI risks are documented and clear."),
        _r("gv-4.3", "GOVERN 4.3", "Testing, incidents and information sharing",
           "Practices are in place to enable AI testing, identification of incidents and information sharing.",
           ("testing_evaluations", "audit_chain_verified")),
        _r("gv-6.1", "GOVERN 6.1", "Third-party AI risks",
           "Policies and procedures address AI risks from third-party entities.",
           ("egress_allowlist", "mcp_change_review")),
        _r("map-3.5", "MAP 3.5", "Human oversight processes",
           "Processes for human oversight are defined, assessed and documented.",
           ("approval_rules",)),
        _r("map-4.1", "MAP 4.1", "Third-party components",
           "Approaches for mapping the risks of AI components, including third-party software, are in place.",
           ("inventory_reviewed", "shadow_ai_discovery")),
        _r("ms-2.4", "MEASURE 2.4", "Monitoring in production",
           "The functionality and behaviour of the AI system are monitored in production.",
           ("usage_observability", "model_monitoring", "prompt_versioning")),
        _r("ms-2.7", "MEASURE 2.7", "Security and resilience",
           "AI system security and resilience are evaluated and documented.",
           ("prompt_injection_protection", "owasp_coverage", "testing_evaluations")),
        _r("ms-2.8", "MEASURE 2.8", "Transparency and accountability",
           "Risks associated with transparency and accountability are examined and documented.",
           ("audit_chain_verified", "agent_identity"),
           "Tamper-evident records show which agent did what, and for whom."),
        _r("ms-2.10", "MEASURE 2.10", "Privacy risk",
           "The privacy risk of the AI system is examined and documented.",
           ("pii_protection_production", "secrets_protection")),
        _r("ms-3.1", "MEASURE 3.1", "Tracking risks over time",
           "Approaches are in place to regularly identify and track existing and emergent AI risks.",
           ("model_monitoring", "mcp_change_review")),
        _r("mg-2.4", "MANAGE 2.4", "Disengage or deactivate",
           "Mechanisms are in place to supersede, disengage or deactivate AI systems that behave outside "
           "their intended use.",
           ("kill_switch_ready",)),
        _r("mg-3.1", "MANAGE 3.1", "Third-party risks monitored",
           "Risks from third-party resources are regularly monitored and controls are applied.",
           ("egress_allowlist", "shadow_ai_discovery", "mcp_change_review")),
        _r("mg-4.1", "MANAGE 4.1", "Post-deployment monitoring",
           "Post-deployment monitoring plans are implemented, including override, decommissioning and "
           "incident response.",
           ("usage_observability", "approval_rules", "kill_switch_ready")),
    ),
)

ISO_42001 = Framework(
    id="iso_42001", title="ISO/IEC 42001", version="ISO/IEC 42001:2023, Annex A",
    source="https://www.iso.org/standard/81230.html",
    requirements=(
        _r("a.2.2", "A.2.2", "AI policy", "The organisation documents a policy for the development or use of AI.",
           note="The organisational AI policy is a document; ZNYX enforces technical policies under it."),
        _r("a.3.2", "A.3.2", "AI roles and responsibilities",
           "Roles and responsibilities for AI are defined and allocated."),
        _r("a.4.2", "A.4.2", "Resource documentation",
           "The resources of AI systems are identified and documented.",
           ("inventory_reviewed",)),
        _r("a.4.4", "A.4.4", "Tooling resources",
           "The tooling resources used for AI systems (models, tools, services) are documented.",
           ("inventory_reviewed", "shadow_ai_discovery")),
        _r("a.5.2", "A.5.2", "AI system impact assessment process",
           "A process assesses the potential consequences of AI systems for individuals and society."),
        _r("a.6.2.4", "A.6.2.4", "AI system verification and validation",
           "Verification and validation measures are defined and documented.",
           ("testing_evaluations", "owasp_coverage")),
        _r("a.6.2.6", "A.6.2.6", "AI system operation and monitoring",
           "The elements needed to operate and monitor the AI system are defined and in place.",
           ("usage_observability", "model_monitoring", "kill_switch_ready", "budgets_set")),
        _r("a.6.2.7", "A.6.2.7", "AI system technical documentation",
           "Technical documentation of the AI system is available to those who need it.",
           ("inventory_reviewed", "policy_published")),
        _r("a.6.2.8", "A.6.2.8", "AI system recording of event logs",
           "Event logs are recorded during the AI system's use.",
           ("audit_chain_verified", "runtime_audit_shipping", "log_retention")),
        _r("a.7.2", "A.7.2", "Data for development and enhancement of AI systems",
           "Data management processes for developing AI systems are defined.",
           note="ZNYX does not train models or hold training data."),
        _r("a.8.2", "A.8.2", "System documentation and information for users",
           "Users are given the information they need about the AI system."),
        _r("a.9.2", "A.9.2", "Processes for responsible use of AI systems",
           "Processes for the responsible use of AI systems are defined and documented.",
           ("policy_published", "approval_rules", "pii_protection_production")),
        _r("a.9.4", "A.9.4", "Intended use of the AI system",
           "The AI system is used according to its intended use and documentation.",
           ("policy_published", "agent_identity", "prompt_versioning")),
        _r("a.10.3", "A.10.3", "Suppliers",
           "Suppliers of AI services, products and components are chosen and monitored in line with the "
           "organisation's approach to responsible AI.",
           ("egress_allowlist", "mcp_change_review", "shadow_ai_discovery")),
    ),
)

FRAMEWORKS: Dict[str, Framework] = {f.id: f for f in (EU_AI_ACT, NIST_AI_RMF, ISO_42001)}


def checks_used() -> List[str]:
    """Every check a framework uses, in CHECKS order."""
    used = {c for f in FRAMEWORKS.values() for r in f.requirements for c in r.checks}
    return [c for c in CHECKS if c in used]


def status_of(requirement: Requirement, results: Mapping[str, Optional[bool]]) -> str:
    if not requirement.checks:
        return "organisational"
    passed = sum(1 for c in requirement.checks if results.get(c) is True)
    if passed == len(requirement.checks):
        return "supported"
    return "partial" if passed else "not_yet"


def assess(framework_id: str, results: Mapping[str, Optional[bool]]) -> Dict[str, object]:
    """The status of each requirement of a framework, given the org's check results. A check
    missing from ``results`` (or None) counts as not evidenced. KeyError for an unknown framework."""
    framework = FRAMEWORKS[framework_id]
    rows = []
    counts = {s: 0 for s in STATUSES}
    for req in framework.requirements:
        status = status_of(req, results)
        counts[status] += 1
        rows.append({"id": req.id, "ref": req.ref, "title": req.title, "summary": req.summary,
                     "note": req.note, "status": status,
                     "checks": [{"check": c, "description": CHECKS[c], "passed": results.get(c)}
                                for c in req.checks]})
    return {"framework": framework.id, "title": framework.title, "version": framework.version,
            "source": framework.source, "disclaimer": DISCLAIMER, "counts": counts, "requirements": rows}


def describe() -> List[Dict[str, object]]:
    """The frameworks and their requirements, without an org's results."""
    return [{"id": f.id, "title": f.title, "version": f.version, "source": f.source,
             "requirements": [{"id": r.id, "ref": r.ref, "title": r.title, "summary": r.summary,
                               "note": r.note, "checks": list(r.checks)} for r in f.requirements]}
            for f in FRAMEWORKS.values()]


def _self_check() -> None:
    for f in FRAMEWORKS.values():
        ids = [r.id for r in f.requirements]
        assert len(ids) == len(set(ids)), f"duplicate requirement id in {f.id}"
        for r in f.requirements:
            unknown = [c for c in r.checks if c not in CHECKS]
            assert not unknown, f"{f.id} {r.id}: unknown checks {unknown}"


_self_check()
