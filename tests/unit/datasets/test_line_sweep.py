import pytest
from urban_sound.datasets.line_sweep import remove_overlap


@pytest.mark.parametrize(
    ("indices", "starts", "ends", "indices_to_delete"),
    [
        (range(5), [0.1, 0.4, 0.3, 1.0, 2.0], [0.25, 0.7, 1.2, 1.5, 4.5], [1, 2, 3]),
        (
            range(10),
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
            range(10),
        ),
        ([1000, 2, 3, 78], [10.0, 15.0, 22.0, 12.5], [11.0, 16.0, 24.0, 14.5], []),
    ],
)
def test_line_sweep(indices, starts, ends, indices_to_delete):
    assert sorted(indices_to_delete) == sorted(remove_overlap(indices, starts, ends))
