"""Wrap deploy/prometheus/rules.yml in a PrometheusRule for the Prometheus Operator."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy/prometheus/rules.yml"
TARGET = ROOT / "deploy/k8s/base/prometheus-rules.yaml"


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str)


def render() -> str:
    groups = yaml.safe_load(SOURCE.read_text())["groups"]
    doc = {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "PrometheusRule",
        "metadata": {"name": "bellwether", "labels": {"release": "kube-prometheus-stack"}},
        "spec": {"groups": groups},
    }
    header = (
        "# Generated from deploy/prometheus/rules.yml by scripts/gen_rules.py (make rules). Do not edit.\n"
    )
    return header + yaml.dump(doc, Dumper=_Dumper, sort_keys=False, width=120)


if __name__ == "__main__":
    TARGET.write_text(render())
    print(f"wrote {TARGET.relative_to(ROOT)}")
