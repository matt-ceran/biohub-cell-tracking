import pandas as pd
import pytest

from biohub.submission import assemble_submission, edges_frame, nodes_frame, validate_submission


def test_valid_submission_passes():
    nodes = nodes_frame(
        "44b6_sample",
        node_ids=[1, 2],
        t=[0, 1],
        z=[32, 33],
        y=[128, 130],
        x=[128, 125],
    )
    edges = edges_frame("44b6_sample", source_ids=[1], target_ids=[2])
    submission = assemble_submission([nodes, edges])

    validate_submission(submission, expected_datasets=["44b6_sample"])
    assert submission["id"].tolist() == [0, 1, 2]


def test_submission_rejects_missing_edge_node():
    nodes = nodes_frame("44b6_sample", node_ids=[1], t=[0], z=[32], y=[128], x=[128])
    edges = edges_frame("44b6_sample", source_ids=[1], target_ids=[99])
    submission = assemble_submission([nodes, edges])

    with pytest.raises(ValueError, match="missing node IDs"):
        validate_submission(submission)


def test_submission_rejects_wrong_columns():
    bad = pd.DataFrame({"id": [0]})
    with pytest.raises(ValueError, match="columns"):
        validate_submission(bad)
