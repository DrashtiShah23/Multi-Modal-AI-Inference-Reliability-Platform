from __future__ import annotations

from pathlib import Path

import pytest

from bellwether.models import build as model_build


@pytest.fixture(scope="session")
def model_repos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build all three releases once per test session, exactly as `make models` does."""
    out = tmp_path_factory.mktemp("models")
    v1 = model_build.make_variant("v1", n_train=12000)
    repos = {}
    for name in model_build.VARIANTS:
        w = v1 if name == "v1" else model_build.make_variant(name, n_train=12000, reference=v1)
        model_build.write_repository(w, out / name, version=model_build.VARIANTS.index(name) + 1)
        repos[name] = out / name
    return repos


@pytest.fixture(scope="session")
def runtime_v1(model_repos):
    from bellwether.models.runtime import MultimodalRuntime

    return MultimodalRuntime(model_repos["v1"])
