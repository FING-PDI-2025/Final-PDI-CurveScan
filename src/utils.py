from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Generator, Self, TypeAlias
from matplotlib.figure import Figure
import cv2
import cv2 as cv
import numpy as np
from rich.console import Console

console = Console()

class Utils:
    @staticmethod
    def console_dump_bgr(image: np.ndarray) -> None:
        """
        Dump the image to the console using rich
        """

        concatenated = ""
        for i in range(0, image.shape[0], 1):
            for j in range(0, image.shape[1], 1):
                b, g, r = image[i, j]
                concatenated += f"[on rgb({r},{g},{b})]  [/on rgb({r},{g},{b})]"
            concatenated += "\n"
        console.print(concatenated)

    @staticmethod
    def show_image(*image: np.ndarray, wait=True, offset=0) -> None:
        """
        Show the image using OpenCV
        """
        for i, img in enumerate(image):
            window = f"Image {i + offset}"
            cv.namedWindow(window, cv2.WINDOW_NORMAL)
            cv.imshow(window, img)
            pos = (i + offset)
            x = pos % 12
            y = pos // 12
            size = 900
            cv.moveWindow(window, size * x + 1, size * y + 1)
            cv.resizeWindow(window, size, size)
        if not wait:
            return
        cv.waitKey(0)
        cv.destroyAllWindows()


def lerp_color(color1, color2, t):
    """
    Linearly interpolate between two colors.
    :param color1: The first color (BGR tuple).
    :param color2: The second color (BGR tuple).
    :param t: Interpolation factor (0.0 to 1.0).
    :return: Interpolated color (BGR tuple).
    """
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))


@cache
def gamma_lut(gamma: float):
    inv_gamma = 1.0 / gamma
    return np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype(np.uint8)


def adjust_gamma(image, gamma=1.0):
    return cv.LUT(image, gamma_lut(gamma))


def smooth_contour(contour, kernel_size=91):
    x, y, w, h = cv.boundingRect(contour)
    padding = kernel_size // 2 + 1
    mask = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.uint8)
    shifted_contour = contour - [x - padding, y - padding]
    cv.drawContours(mask, [shifted_contour], -1, 255, thickness=cv.FILLED)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv.blur(mask, (kernel_size, kernel_size))
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        new_contour = contours[0] + [x - padding, y - padding]
        new_area = cv.contourArea(new_contour)
        if len(contours) != 1:
            print(
                f"Smoothing contours caused it to split into {len(contours)} contours"
            )
        return new_contour
    else:
        print("Failed to smooth.")
        return contour


def file_cache(base: str, key: str, compute: Callable[[], Any]) -> Any:
    """
    Cache the result of a function call based on the dataset item and key.
    """
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{base}_{key}.npy"
    if cache_file.exists():
        return np.load(cache_file, allow_pickle=True)
    else:
        value = compute()
        np.save(cache_file, value)
        return value


@dataclass
class ImageProcessingIntermediateStepReporter:
    output_dir: Path
    output_count: int = 0

    indent: str = ""

    def output(self, title: str, image: np.ndarray, kind: str = "step") -> Path:
        """
        Save the intermediate step image to the output directory.
        """
        output_path = self.output_dir / self.indent / f"{kind}_{self.output_count:03d}_{title}.jpeg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(output_path), image)
        self.output_count += 1
        console.print(f"Saved intermediate step {self.output_count}, {title} to {output_path.absolute()}")
        return output_path

    @contextmanager
    def with_indent(self, indent: str) -> Generator[Self]:
        """
        Create a new reporter with the given indent.
        """
        old_indent = self.indent
        self.indent += indent
        try:
            yield self
        finally:
            self.indent = old_indent

    def output_mpl(self, title: str, fig: Figure, kind: str = "step") -> Path:
        """
        Save the intermediate step matplotlib figure to the output directory.
        """
        output_path = self.output_dir / self.indent / f"{kind}_{self.output_count:03d}_{title}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        self.output_count += 1
        console.print(f"Saved intermediate step {self.output_count}, {title} to {output_path.absolute()}")
        return output_path


Img: TypeAlias = cv.Mat | np.ndarray[Any, np.dtype]
Contour: TypeAlias = Img
