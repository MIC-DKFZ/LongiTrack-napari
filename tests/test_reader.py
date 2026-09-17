
import numpy as np
import pytest
from synthetic import SPACING_ZYX, write_sample_pair

from longitrack_napari._reader import napari_get_reader, read_volume


@pytest.fixture(scope="module")
def sample_pair(tmp_path_factory):
    return write_sample_pair(tmp_path_factory.mktemp("sample"))


def test_get_reader_accepts_known_suffixes():
    assert napari_get_reader("scan.nii.gz") is not None
    assert napari_get_reader(["a.nii.gz", "b.mha"]) is not None
    assert napari_get_reader("notes.txt") is None
    assert napari_get_reader([]) is None


def test_get_reader_rejects_mixed_lists():
    assert napari_get_reader(["scan.nii.gz", "notes.txt"]) is None


def test_read_volume_matches_simpleitkio(sample_pair):
    from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO

    baseline, _ = sample_pair
    ours, kwargs = read_volume(baseline)
    theirs, properties = SimpleITKIO().read_images([str(baseline)])

    assert np.array_equal(ours, theirs[0])
    assert tuple(kwargs["scale"]) == tuple(properties["spacing"])
    assert tuple(kwargs["scale"]) == SPACING_ZYX

