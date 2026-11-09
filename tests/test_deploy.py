"""Consistency checks between code and deployment config, so they cannot drift apart silently."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def _gateway_flags() -> set[str]:
    src = (ROOT / "gateway/cmd/gateway/main.go").read_text()
    return set(re.findall(r'fs\.\w+Var\(&o\.\w+, "([a-z-]+)"', src))


@pytest.mark.parametrize(
    "path", sorted(p for p in DEPLOY.rglob("*.y*ml")), ids=lambda p: str(p.relative_to(ROOT))
)
def test_yaml_parses(path):
    assert _docs(path)


def test_prometheus_rule_is_generated_from_rules_file():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen_rules", ROOT / "scripts/gen_rules.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (DEPLOY / "k8s/base/prometheus-rules.yaml").read_text() == mod.render(), "run `make rules`"


def test_every_metric_in_rules_is_exported():
    exprs = " ".join(
        r.get("expr", "")
        for g in yaml.safe_load((DEPLOY / "prometheus/rules.yml").read_text())["groups"]
        for r in g["rules"]
    )
    exprs += (DEPLOY / "k8s/base/analysis-template.yaml").read_text()
    used = {re.sub(r"_(bucket|sum|count)$", "", m) for m in re.findall(r"\bbellwether_[a-z_]+", exprs)}
    sources = (ROOT / "gateway/internal/proxy/proxy.go").read_text() + (
        ROOT / "src/bellwether/analyzer/telemetry.py"
    ).read_text()
    missing = {m for m in used if f'"{m}"' not in sources}
    assert not missing, f"rules reference metrics nobody exports: {missing}"


def test_runbook_covers_every_alert():
    runbook = (ROOT / "docs/runbook.md").read_text().lower()
    anchors = set()
    for group in yaml.safe_load((DEPLOY / "prometheus/rules.yml").read_text())["groups"]:
        for rule in group["rules"]:
            if "alert" in rule:
                anchors.add(rule["annotations"]["runbook"].split("#")[1])
    headings = {
        re.sub(r"[^a-z0-9 -]", "", h).strip().replace(" ", "-")
        for h in re.findall(r"^#+ (.+)$", runbook, re.M)
    }
    assert anchors <= headings, f"runbook is missing sections: {anchors - headings}"


def test_gateway_args_are_real_flags():
    flags = _gateway_flags()
    assert {"primary-url", "candidate-url", "analyzer-url", "mirror-fraction"} <= flags
    k8s = next(d for d in _docs(DEPLOY / "k8s/base/gateway.yaml") if d["kind"] == "Deployment")
    args = k8s["spec"]["template"]["spec"]["containers"][0]["args"]
    compose = yaml.safe_load((DEPLOY / "compose/docker-compose.yaml").read_text())["services"]["gateway"][
        "command"
    ]
    for arg in args + compose:
        assert arg.lstrip("-").split("=")[0] in flags, arg


def test_analysis_template_targets_the_verdict_route():
    from bellwether.analyzer.api import create_app

    template = _docs(DEPLOY / "k8s/base/analysis-template.yaml")[0]
    url = template["spec"]["metrics"][0]["provider"]["web"]["url"]
    path = re.sub(r"\{\{args\.[a-z-]+\}\}", "X", url.split(":8010", 1)[1])

    class _Stub:  # create_app only registers routes
        pass

    routes = {r.path for r in create_app(_Stub()).routes}
    pattern = {re.sub(r"\{[a-z_]+\}", "X", r) for r in routes}
    assert path in pattern


def test_rollout_wiring():
    docs = _docs(DEPLOY / "k8s/base/modelserver.yaml")
    rollout = next(d for d in docs if d["kind"] == "Rollout")
    services = {d["metadata"]["name"] for d in docs if d["kind"] == "Service"}
    canary = rollout["spec"]["strategy"]["canary"]
    assert {canary["stableService"], canary["canaryService"]} <= services
    analysis = next(s["analysis"] for s in canary["steps"] if "analysis" in s)
    assert (
        analysis["templates"][0]["templateName"]
        == _docs(DEPLOY / "k8s/base/analysis-template.yaml")[0]["metadata"]["name"]
    )
    gateway = next(d for d in _docs(DEPLOY / "k8s/base/gateway.yaml") if d["kind"] == "Deployment")
    args = " ".join(gateway["spec"]["template"]["spec"]["containers"][0]["args"])
    assert f"//{canary['stableService']}:" in args and f"//{canary['canaryService']}:" in args


def test_kustomization_resources_exist():
    for kfile in DEPLOY.rglob("kustomization.yaml"):
        k = yaml.safe_load(kfile.read_text())
        for res in k.get("resources", []):
            assert (kfile.parent / res).exists(), f"{kfile}: {res}"
        for patch in k.get("patches", []):
            if "path" in patch:
                assert (kfile.parent / patch["path"]).exists()
