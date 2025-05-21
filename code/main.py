import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
from typing import Optional
import cv2 as cv
from typing import Self
import pandas as pd
import logging
from rich.layout import Panel
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
from rich.console import Console

console = Console()


class Dataset:
    path: Path = Path(__file__).parent.parent / "datasets" / "caratulas"

    def __init__(self):
        self.images = [image for image in self.path.glob("*.jpeg")]
        log.debug(f"Found {len(self.images)} images.")

    @cached_property
    def items(self) -> list["DatasetItem"]:
        """
        Get the items in the dataset
        """
        return [DatasetItem(image, self) for image in self.images]

    @cached_property
    def df(self) -> pd.DataFrame:
        """
        Create a dataframe with the images and their properties
        """
        data = []
        for item in self.items:
            data.append(
                {
                    "name": item.name,
                    "path": item.path,
                    "has_flash": item.has_flash,
                    "has_light": item.has_light,
                }
            )
        return (
            pd.DataFrame(data)
            .sort_values(["name", "has_flash", "has_light"])
            .reset_index(drop=True)
        )

    @cached_property
    def pretty_df(self) -> pd.DataFrame:
        df = self.df.copy(deep=True)
        df["path"] = df["path"].apply(lambda x: x.name)
        return df

    def get(
        self, name: str, has_flash: bool = True, has_light: bool = True
    ) -> Optional["DatasetItem"]:
        """
        Get the image with the given name and properties
        """
        for item in self.items:
            if (
                item.name == name
                and item.has_flash == has_flash
                and item.has_light == has_light
            ):
                return item
        return None


@dataclass
class DatasetItem:
    path: Path
    _base_dataset: Dataset

    @cached_property
    def name(self) -> str:
        """The name of the image without [flash] or [light] terms"""
        stem = self.path.stem
        parts = stem.split("_")
        for column in ["flash", "light"]:
            if "no" + column in parts:
                parts.remove("no" + column)
            if column in parts:
                parts.remove(column)
        return "_".join(parts)

    @cached_property
    def has_flash(self) -> bool:
        if not "flash" in self.path.stem.lower():
            raise ValueError(f"Image {self.path} does not have flash in its name")
        return "noflash" not in self.path.stem.lower()

    @cached_property
    def has_light(self) -> bool:
        if not "light" in self.path.stem.lower():
            raise ValueError(f"Image {self.path} does not have light in its name")
        return "nolight" not in self.path.stem.lower()

    @property
    def with_flash(self) -> Self:
        """
        Get the image with flash
        """
        return self._base_dataset.get(
            self.name, has_flash=True, has_light=self.has_light
        )

    @property
    def with_light(self) -> Self:
        """
        Get the image with light
        """
        return self._base_dataset.get(
            self.name, has_flash=self.has_flash, has_light=True
        )

    @property
    def without_flash(self) -> Self:
        """
        Get the image without flash
        """
        return self._base_dataset.get(
            self.name, has_flash=False, has_light=self.has_light
        )

    @property
    def without_light(self) -> Self:
        """
        Get the image without light
        """
        return self._base_dataset.get(
            self.name, has_flash=self.has_flash, has_light=False
        )

    @property
    def other_variants(self) -> list[Self]:
        """
        Get the other variants of the image
        """
        return [
            item
            for item in self._base_dataset.items
            if item.name == self.name and item.path != self.path
        ]

    @cached_property
    def data(self) -> np.ndarray:
        """
        Get the image data
        """
        return cv.imread(str(self.path.absolute()), cv.IMREAD_UNCHANGED)


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
    def show_image(*image: np.ndarray) -> None:
        """
        Show the image using OpenCV
        """
        for i, img in enumerate(image):
            window = f"Image {i}"
            cv.namedWindow(window, cv2.WINDOW_NORMAL)
            cv.imshow(window, img)
            cv.moveWindow(window, 800 * i, 100)
            cv.resizeWindow(window, 800, 600)
        cv.waitKey(0)
        cv.destroyAllWindows()


def border_detect(image: np.ndarray) -> np.ndarray:
    """
    Detect the border of the image
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    return edges


def line_detect(image: np.ndarray) -> np.ndarray:
    """
    Detect the lines of the image
    """
    # Convert to grayscale and apply Canny edge detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def region_detect(image: np.ndarray) -> np.ndarray:
    """
    Detect the regions of the image
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv.drawContours(image, [contour], -1, (0, 255, 0), 2)
    return image


def region_detect2(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect the regions of the image based on Canny edges
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    highest_area_contour = max(contours, key=cv.contourArea)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask = cv.drawContours(
        mask, [highest_area_contour], -1, (255, 255, 255), thickness=cv.FILLED
    )
    # mask = cv.drawContours(mask, [highest_area_contour], -1, (0, 0, 0), thickness=2)
    # apply top-hat transformation
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    # Clean up the mask
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    area = cv.contourArea(highest_area_contour)
    image = cv.bitwise_and(image, image, mask=mask)

    return image, area


def is_image_blurry(image: np.ndarray) -> float:
    """
    Check if the image is blurry using the Laplacian variance method
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
    return laplacian_var

def is_image_blurry2(image: np.ndarray) -> float:
    """
    Check if the image is blurry using the Canny edge detection method
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    return edge_count
def is_image_blurry3(image: np.ndarray) -> float:
    """
    Check if the image is blurry using Fourier transform
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
    mean_magnitude = np.mean(magnitude_spectrum)
    return mean_magnitude

def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the image using a kernel
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened

def low_pass_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a low pass filter to the image
    """
    kernel = np.ones((5, 5), np.float32) / 25
    filtered = cv.filter2D(image, -1, kernel)
    return filtered

def high_pass_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a high pass filter to the image
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filtered = cv.filter2D(image, -1, kernel)
    return filtered

def main():
    dataset = Dataset()
    # log.info(dataset.pretty_df)
    sample = dataset.get("simple", False, False)
    log.info(sample)
    df_rows = []
    for smp in dataset.items:
        blurry = is_image_blurry3(smp.data)
        is_blurry = blurry < 50
        image_data = smp.data
        # if is_blurry:
        #     image_data = sharpen(image_data)
        #     image_data = sharpen(image_data)

        processed, area = region_detect2(image_data)
        estimated_is_detection_correct = area > 100_000
        if not estimated_is_detection_correct:
            image_data = high_pass_filter(image_data)
            processed, area = region_detect2(image_data)
            estimated_is_detection_correct = area > 100_000
            # Utils.show_image(image_data, processed)
        df_rows.append(
            {
                "name": smp.name,
                "has_flash": smp.has_flash,
                "has_light": smp.has_light,
                "blur": blurry,
                "is_blurry": is_blurry,
                "area": area,
                "is_correct (estimated)": estimated_is_detection_correct,
            }
        )
        # console.print(
        #     Panel(
        #         f"Image: {smp.name}\n"
        #         f"Has flash: {smp.has_flash}\n"
        #         f"Has light: {smp.has_light}\n"
        #         f"Is blurry: {is_image_blurry(smp.data)}\n"
        #         f"Detected area: {area}\n"
        #         f"Is correct (estimated): {estimated_is_detection_correct}",
        #         highlight=True,
        #         title=smp.path.name,
        #     )
        # )
    df = pd.DataFrame(df_rows)
    df = df.sort_values(["is_correct (estimated)", "blur", "name", "has_flash", "has_light"]).reset_index(drop=True)
    console.print(Panel(str(df), highlight=True))
    console.print(f"Total correct detections (estimated): {df['is_correct (estimated)'].sum()}")



if __name__ == "__main__":
    main()
