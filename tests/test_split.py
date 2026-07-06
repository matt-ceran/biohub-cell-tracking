from biohub.split import make_embryo_holdout


def test_make_embryo_holdout_keeps_embryos_disjoint():
    split = make_embryo_holdout(
        [
            "44b6_a",
            "44b6_b",
            "6bba_a",
            "6bba_b",
            "99aa_a",
        ],
        validation_fraction=0.34,
        seed=1,
    )
    assert set(split.train_embryos).isdisjoint(split.validation_embryos)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train) | set(split.validation) == {
        "44b6_a",
        "44b6_b",
        "6bba_a",
        "6bba_b",
        "99aa_a",
    }
