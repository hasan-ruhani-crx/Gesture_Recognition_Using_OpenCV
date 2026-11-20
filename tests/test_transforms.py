import numpy as np

from src.capture.transforms import (
    bgr_to_gray,
    bgr_to_rgb,
    flip_horizontal,
    resize_to_width,
)


def test_flip_horizontal() -> None:
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]

    flipped = flip_horizontal(img)

    assert flipped.shape == (4, 6, 3)
    assert np.array_equal(flipped[0, 5], [10, 20, 30])
    assert np.array_equal(flipped[0, 0], [0, 0, 0])
    assert flipped.flags["C_CONTIGUOUS"] is True


def test_bgr_to_rgb() -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [10, 20, 30]  # B=10, G=20, R=30

    rgb = bgr_to_rgb(img)

    assert np.array_equal(rgb[0, 0], [30, 20, 10])
    assert rgb.flags["C_CONTIGUOUS"] is True


def test_bgr_to_gray_bt601() -> None:
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = [100, 150, 200]  # B=100, G=150, R=200

    gray = bgr_to_gray(img)
    expected = round(0.299 * 200 + 0.587 * 150 + 0.114 * 100)  # 159

    assert gray.shape == (1, 1)
    assert gray.dtype == np.uint8
    assert abs(int(gray[0, 0]) - expected) <= 1


def test_resize_to_width() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    downscaled = resize_to_width(img, target_width=100)
    assert downscaled.shape == (50, 100, 3)

    upscaled = resize_to_width(img, target_width=400)
    assert upscaled.shape == (200, 400, 3)

    same = resize_to_width(img, target_width=200)
    assert same is img  # Fast path avoids unnecessary copy

    try:
        resize_to_width(img, target_width=0)
        raise AssertionError("Expected ValueError for non-positive target_width")
    except ValueError:
        pass


if __name__ == "__main__":
    test_flip_horizontal()
    test_bgr_to_rgb()
    test_bgr_to_gray_bt601()
    test_resize_to_width()
    print("All transform unit tests passed.")