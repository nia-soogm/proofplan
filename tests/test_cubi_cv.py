"""Synthetic checks for the CubiCasa classical-CV pieces (no dataset needed)."""
import numpy as np
import cv2
from bp.sources import cubi_cv, cubi_door
from bp.sources.cubi_sym import nms_count


def _plan(img, x0, y0, x1, y1):
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 6)
    cv2.line(img, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (0, 0, 0), 3)
    cv2.line(img, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2), (0, 0, 0), 3)


def test_two_drawings_side_by_side_are_two_parts(tmp_path):
    img = np.full((500, 1000, 3), 255, np.uint8)
    _plan(img, 40, 60, 440, 440)
    _plan(img, 560, 60, 960, 440)
    cv2.line(img, (40, 470), (960, 470), (0, 0, 0), 1)  # a dimension line bridging the gap
    p = tmp_path / "two.png"; cv2.imwrite(str(p), img)
    assert len(cubi_cv.big_parts(str(p), 0.08)[0]) == 2
    one = np.full((500, 1000, 3), 255, np.uint8); _plan(one, 40, 60, 960, 440)
    q = tmp_path / "one.png"; cv2.imwrite(str(q), one)
    assert len(cubi_cv.big_parts(str(q), 0.08)[0]) == 1


def test_door_swing_is_detected(tmp_path):
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.line(img, (20, 200), (280, 200), (0, 0, 0), 4)  # wall
    hinge, r = (100, 200), 40  # 5 label heights at s=8
    cv2.line(img, hinge, (hinge[0], hinge[1] - r), (0, 0, 0), 2)  # open leaf
    cv2.ellipse(img, hinge, (r, r), 0, 270, 360, (0, 0, 0), 1)  # swing arc
    p = tmp_path / "door.png"; cv2.imwrite(str(p), img)
    hits = cubi_door.detect(str(p), 8.0, *cubi_door.LOOSE)
    assert any(x0 <= hinge[0] + r // 2 <= x1 and y0 <= hinge[1] - r // 2 <= y1 for x0, y0, x1, y1, _ in hits)


def test_nms_merges_double_bowl():
    bx = np.array([[0, 0, 10, 10], [11, 0, 21, 10], [100, 100, 110, 110]])
    sc = np.array([0.9, 0.8, 0.95])
    assert len(nms_count(bx, sc, 0.5, merge=1.6)) == 2
    assert len(nms_count(bx, sc, 0.5, merge=0.5)) == 3
