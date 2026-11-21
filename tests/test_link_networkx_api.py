"""Regression guard for the networkx-backed flow linker and track pruning.

NetworkX 3.7 changes some of its APIs (for example ``maximal_independent_set`` now
returns a set), so this test pins the observable output of the two code paths that
actually use networkx. A future dependency bump must not silently change the tracks
the pipeline emits.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from biohub.link import link_graph_flow, prune_to_tracks
from biohub.metric import TrackingGraph

UNIT_SCALE = {"z": 1.0, "y": 1.0, "x": 1.0}


def _networkx_requirements() -> list[str]:
    """Return the project's declared networkx requirements."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return [
        dep
        for dep in data["project"]["dependencies"]
        if dep.split(";", 1)[0].strip().startswith("networkx")
    ]


def test_pyproject_requires_networkx_37_where_supported():
    """Python >= 3.12 installs must resolve to the networkx 3.7 API release.

    networkx 3.7 itself requires Python >= 3.12, so the 3.11 branch keeps the latest
    release that still supports the project's declared minimum interpreter.
    """
    requirements = _networkx_requirements()
    python_312 = [
        req for req in requirements if ";" not in req or "python_version >= '3.12'" in req
    ]
    assert any(">=3.7" in req for req in python_312)


def _nodes(rows):
    """Build a node-only TrackingGraph from (node_id, t, z, y, x) rows."""
    rows = list(rows)
    return TrackingGraph(
        node_ids=np.array([r[0] for r in rows], dtype=np.int64),
        t=np.array([r[1] for r in rows]),
        z=np.array([r[2] for r in rows], dtype=float),
        y=np.array([r[3] for r in rows], dtype=float),
        x=np.array([r[4] for r in rows], dtype=float),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_networkx_flow_linking_and_pruning_keep_the_expected_track():
    """The min-cost-flow linker still chains a drifting cell, and pruning still drops
    an isolated detection without disturbing that track."""
    nodes = _nodes(
        [
            (1, 0, 0.0, 10.0, 10.0),  # track start
            (2, 1, 0.0, 12.0, 10.0),  # continuation
            (3, 2, 0.0, 14.0, 10.0),  # track end
            (99, 1, 0.0, 200.0, 200.0),  # isolated junk detection
        ]
    )

    linked = link_graph_flow(nodes, scale=UNIT_SCALE)
    assert {tuple(edge) for edge in linked.edges} == {(1, 2), (2, 3)}

    pruned = prune_to_tracks(linked, min_track_length=2)
    assert set(pruned.node_ids) == {1, 2, 3}
    assert {tuple(edge) for edge in pruned.edges} == {(1, 2), (2, 3)}
