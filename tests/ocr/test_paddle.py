import numpy as np
from numpy.testing import assert_array_equal

from e2p2.ocr.paddle import sorted_boxes


def test_sorted_boxes():
    text_boxes = np.array(
        [
            [[403.0, 396.0], [1204.0, 398.0], [1204.0, 434.0], [402.0, 433.0]],
            [[441.0, 174.0], [1166.0, 176.0], [1165.0, 222.0], [441.0, 221.0]],
            [[403.0, 346.0], [1204.0, 348.0], [1204.0, 384.0], [402.0, 383.0]],
        ]
    )

    sorted_text_boxes = sorted_boxes(text_boxes)

    assert_array_equal(sorted_text_boxes[0], text_boxes[1])
    assert_array_equal(sorted_text_boxes[1], text_boxes[2])
    assert_array_equal(sorted_text_boxes[2], text_boxes[0])
