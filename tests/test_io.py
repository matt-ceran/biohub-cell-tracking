from pathlib import Path

from biohub.io import dataset_name, discover_split, embryo_id, read_image_metadata


def test_dataset_and_embryo_names():
    assert dataset_name("44b6_0113de3b.zarr") == "44b6_0113de3b"
    assert dataset_name(Path("44b6_0113de3b.geff")) == "44b6_0113de3b"
    assert embryo_id("44b6_0113de3b") == "44b6"


def test_discover_split_and_read_metadata(tmp_path):
    train = tmp_path / "train"
    zarr_dir = train / "44b6_0113de3b.zarr" / "0"
    geff_dir = train / "44b6_0113de3b.geff"
    zarr_dir.mkdir(parents=True)
    geff_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text('{"shape":[100,64,256,256],"data_type":"uint16"}')

    samples = discover_split(tmp_path, "train")
    assert len(samples) == 1
    assert samples[0].dataset == "44b6_0113de3b"
    assert samples[0].labels == geff_dir

    metadata = read_image_metadata(samples[0].image)
    assert metadata["shape"] == [100, 64, 256, 256]
