import itertools

import cv2
import numpy as np
from pathlib import Path
from typing import Any, TypeAlias
import cv2 as cv
import pandas as pd
import logging
from rich.layout import Panel
from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console

from dataset import Dataset
from utils import Utils

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

console = Console()
Img: TypeAlias = cv.Mat | np.ndarray[Any, np.dtype]
Contour: TypeAlias = Img


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
    out = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


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


def region_detect2(image: np.ndarray) -> tuple[Img, float, Contour, Contour]:
    """
    Detect the regions of the image based on Canny edges
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 250)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError(f"No contours found in image.")

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

    # Find contours in the mask
    clean_contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not clean_contours:
        largest_clean_contour = highest_area_contour
    else:
        # Get largest contour
        largest_clean_contour = max(clean_contours, key=cv.contourArea)

    return image, area, highest_area_contour, largest_clean_contour


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
    magnitude_spectrum = 20 * np.log(
        cv.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
    )
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


def find_contour(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Find contours in the image
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    highest_area_contour = max(contours, key=cv.contourArea)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask = cv.drawContours(
        mask, [highest_area_contour], -1, (255, 255, 255), thickness=cv.FILLED
    )
    area = cv.contourArea(highest_area_contour)
    image = cv.bitwise_and(image, image, mask=mask)
    return image, area


def process_samples(dataset, output_dir):
    for smp in dataset.items:
        yield from process_sample(smp, output_dir)


def analyse_results(processed, area, contour, clean_contour):
    """
    Analyse the results of the processed image
    """
    # Detect corners in the contour
    approx = cv2.approxPolyN(clean_contour, 4)
    corners = cv.drawContours(
        np.zeros_like(processed), [approx], -1, (255, 255, 255), thickness=cv.FILLED
    )

    # Count number of non-black pixels in the processed image
    non_black_corners = np.count_nonzero(corners)
    non_black_processed = np.count_nonzero(processed)
    size_ratio = abs(non_black_corners / (non_black_processed + 1) - 1)

    is_correct = area > 100_000 and area < 1_000_000 and size_ratio < 0.1

    return processed, area, contour, clean_contour, is_correct, size_ratio, corners


def process_sample(smp, output_dir: Path):
    try:
        blurry = is_image_blurry3(smp.data)
        is_blurry = blurry < 50
        image_data = smp.data
        (
            raw_processed,
            raw_area,
            raw_contour,
            raw_clean_contour,
            raw_is_correct,
            raw_size_ratio,
            raw_corners,
        ) = analyse_results(*region_detect2(image_data))
        # (
        #     morph_r_processed,
        #     morph_r_area,
        #     morph_r_contour,
        #     morph_r_clean_contour,
        #     morph_r_is_correct,
        #     morph_r_size_ratio,
        #     morph_r_corners,
        # ) = analyse_results(*morphological_region_detect(image_data, 2))
        # (
        #     morph_g_processed,
        #     morph_g_area,
        #     morph_g_contour,
        #     morph_g_clean_contour,
        #     morph_g_is_correct,
        #     morph_g_size_ratio,
        #     morph_g_corners,
        # ) = analyse_results(*morphological_region_detect(image_data, 1))
        (
            morph_b_processed,
            morph_b_area,
            morph_b_contour,
            morph_b_clean_contour,
            morph_b_is_correct,
            morph_b_size_ratio,
            morph_b_corners,
        ) = analyse_results(*morphological_region_detect(image_data, 0))

        # precedence = ["morph_b", "morph_g", "morph_r", "raw"]
        precedence = ["morph_b", "raw"]
        processed = None
        area = 0
        contour = None
        clean_contour = None
        is_correct = False
        size_ratio = 0
        corners = None
        # Select first that "is_correct"
        for name in precedence:
            if locals()[f"{name}_is_correct"]:
                processed = locals()[f"{name}_processed"]
                area = locals()[f"{name}_area"]
                contour = locals()[f"{name}_contour"]
                clean_contour = locals()[f"{name}_clean_contour"]
                is_correct = locals()[f"{name}_is_correct"]
                size_ratio = locals()[f"{name}_size_ratio"]
                corners = locals()[f"{name}_corners"]
                break
        if processed is None:
            # Show morph_b
            processed = morph_b_processed
            area = morph_b_area
            contour = morph_b_contour
            clean_contour = morph_b_clean_contour
            is_correct = morph_b_is_correct
            size_ratio = morph_b_size_ratio
            corners = morph_b_corners

        # log.debug(f"Morph: {morph_non_black_corners / (morph_non_black_processed + 1)}\n" + f"Raw: {raw_non_black_corners / (raw_non_black_processed + 1)}")
        display = np.hstack((image_data, processed, corners))
        # log.debug(display.shape)

        df_row = {
            "name": smp.name,
            "has_flash": smp.has_flash,
            "has_light": smp.has_light,
            "blur": blurry,
            "is_blurry": is_blurry,
            "area": area,
            "is_correct (estimated)": is_correct,
        }
        # Ignore horizontal images.
        yield display if display.shape[0] > 899 else None, df_row
        Utils.show_image(display)
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

           
        # Save processed image
        if processed is not None:
            processed_path = output_dir / f"{smp.path.stem}_processed.jpeg"
            cv.imwrite(str(processed_path), display)

        # Save mask if it is not None
        if processed is not None:
            mask_path = output_dir / f"{smp.path.stem}_mask.png"
            cv.imwrite(str(mask_path), corners)

    except ValueError:
        yield None, {}


def morphological_region_detect(
    image_data: np.ndarray, color: int = 0
) -> tuple[Img, float, Contour, Contour]:
    morphological_gradient: np.ndarray = cv.morphologyEx(
        image_data, cv.MORPH_GRADIENT, cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    )
    yuv = cv.cvtColor(morphological_gradient, cv.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv.equalizeHist(yuv[:, :, 0])
    eq = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
    # Utils.show_image(eq)
    signed_mg: np.ndarray = morphological_gradient.copy().astype(np.int16)
    mask: np.ndarray = mask_blue(signed_mg, color)
    # Find contours in the morphological gradient
    contours, _ = cv.findContours(
        cv.cvtColor(mask, cv.COLOR_BGR2GRAY), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        log.warning("No contours found in the morphological gradient.")
        return mask, 0, [], np.zeros_like(image_data)
    largest_contour = max(contours, key=cv.contourArea)
    # Draw the largest contour on the original image
    contour_render = cv.drawContours(
        np.zeros_like(mask), [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED
    )
    contour_render = cv.cvtColor(contour_render, cv.COLOR_BGR2GRAY)
    contour_render = cv.medianBlur(contour_render, 5)
    # Apply morphological operations to clean up the contours
    contour_render = cv.morphologyEx(
        contour_render,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)),
    )
    # contour_render = cv.morphologyEx(contour_render, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50)))
    area = cv.contourArea(largest_contour)
    masked_image = cv.bitwise_and(image_data, image_data, mask=contour_render)

    # Detect contour on contour_render
    clean_contours, _ = cv.findContours(
        contour_render, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if not clean_contours:
        largest_clean_contour = largest_contour
    else:
        # Get largest contour
        largest_clean_contour = max(clean_contours, key=cv.contourArea)

    return masked_image, area, largest_contour, largest_clean_contour


def mask_blue(inp: np.ndarray, color: int = 0) -> np.ndarray:
    # Detect "Blue"
    match color:
        case 0:
            mask = 2 * inp[:, :, 0] - inp[:, :, 1] - inp[:, :, 2]
        case 1:
            mask = 2 * inp[:, :, 1] - inp[:, :, 0] - inp[:, :, 2]
        case 2:
            mask = 2 * inp[:, :, 2] - inp[:, :, 0] - inp[:, :, 1]

    mask[mask < 0] = 0
    mask = mask.astype(np.uint8)
    mask = cv.medianBlur(mask, 5)
    # Apply morphological operations to clean up the mask
    mask = cv.morphologyEx(
        mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv.morphologyEx(
        mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    )
    mask *= 3
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    return mask


def main():
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    dataset = Dataset()
    render = False
    hide_correct = False
    wait_single = True
    batch_size = 4
    log.info(dataset.pretty_df)
    df_rows = []
    proced = process_samples(dataset, output_dir)
    for i, item in enumerate(
        itertools.batched(
            (
                it
                for it in track(proced, total=len(dataset.items), console=console)
                if not hide_correct or not it[1]["is_correct (estimated)"]
            ),
            batch_size,
        )
    ):
        batch = [x for x, y in item if x is not None]
        df_rows.extend([y for x, y in item])
        if not render:
            continue
        if len(batch) != 1:
            batch = np.vstack(batch)
        else:
            batch = batch[0]
        Utils.show_image(batch, wait=wait_single, offset=0)
    df = pd.DataFrame(df_rows)
    # df = df.sort_values(["is_correct (estimated)", "blur", "name", "has_flash", "has_light"]).reset_index(drop=True)
    console.print(Panel(str(df), highlight=True, expand=False))
    console.print(
        f"Total correct detections (estimated): {df['is_correct (estimated)'].sum()}"
    )
    # Is correct group by (has_flash and has_light)
    correct_grouped = (
        df.groupby(["has_flash", "has_light"])["is_correct (estimated)"]
        .agg(["sum", "count"])
        .reset_index()
    )
    correct_grouped["percentage"] = (
        correct_grouped["sum"] / correct_grouped["count"]
    ) * 100
    console.print(
        Panel(
            str(correct_grouped),
            highlight=True,
            title="Correct Detections Grouped by Flash and Light",
            expand=False,
        )
    )

    correct_grouped = (
        df.groupby(["name"])["is_correct (estimated)"]
        .agg(["sum", "count"])
        .reset_index()
    )
    correct_grouped["percentage"] = (
        correct_grouped["sum"] / correct_grouped["count"]
    ) * 100
    console.print(
        Panel(
            str(correct_grouped),
            highlight=True,
            title="Correct Detections Grouped by Name",
            expand=False,
        )
    )

    correct_total = df["is_correct (estimated)"].sum()
    total_images = len(df)
    console.print(
        f"Total correct detections (estimated): {correct_total} out of {total_images} ({(correct_total / total_images) * 100:.2f}%)"
    )

    if render and not wait_single:
        Utils.show_image(wait=True)


if __name__ == "__main__":
    main()
