import itertools
from functools import cache

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Callable, TypeAlias
import cv2 as cv
import pandas as pd
import logging
from rich.layout import Panel
from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console

from dataset import Dataset
from dataset_item import DatasetItem
from utils import Utils

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")
pd.set_option("display.max_rows", 500)

console = Console()
Img: TypeAlias = cv.Mat | np.ndarray[Any, np.dtype]
Contour: TypeAlias = Img


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
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values

    # apply gamma correction using the lookup table
    return cv.LUT(image, gamma_lut(gamma))


def smooth_contour(contour, kernel_size=91, epsilon=0.01):
    old_area = cv.contourArea(contour)

    # Compute tight bounding rectangle
    x, y, w, h = cv.boundingRect(contour)
    padding = kernel_size // 2 + 1

    # Create small mask based on bounding rect
    mask = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.uint8)

    # Shift contour to bounding rect coordinates
    shifted_contour = contour - [x - padding, y - padding]

    # Draw filled contour on the small mask
    cv.drawContours(mask, [shifted_contour], -1, 255, thickness=cv.FILLED)

    # Efficient smoothing operation
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv.blur(mask, (kernel_size, kernel_size))
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)

    # Find contours on the processed mask
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


def pre_process_image(image: np.ndarray) -> np.ndarray:
    # Morphology
    # blurred = cv.GaussianBlur(image.copy(), (5, 5), 0)
    blurred = cv.medianBlur(image, 5)
    # Apply morphological gradient
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gradient = cv.morphologyEx(blurred, cv.MORPH_GRADIENT, kernel)
    gray_gradient = cv.cvtColor(gradient, cv.COLOR_BGR2GRAY)
    gray_gradient_eq = cv.equalizeHist(gray_gradient)  # Highlight borders even more
    gray_gradient = cv.addWeighted(gray_gradient_eq, 0.15, gray_gradient, 6, 0)
    gray_gradient = adjust_gamma(gray_gradient, 0.60)
    # gray_gradient_eq = cv.equalizeHist(gray_gradient) # Highlight borders even more
    # gray_gradient = cv.addWeighted(gray_gradient_eq, 0.15, gray_gradient, 0.85, 0)
    # s(gray_gradient)
    # Sharpen image
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_OPEN, kernel)
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_CLOSE, kernel)
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_ERODE, kernel)
    gray_gradient = cv.GaussianBlur(gray_gradient, (5, 5), 1)
    # gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_GRADIENT, kernel)

    pre_threshold_gg = gray_gradient.copy()

    # Otsu threshold
    _, gray_gradient = cv.threshold(
        gray_gradient, -1, 255, cv.THRESH_OTSU | cv.THRESH_BINARY
    )
    gray_gradient = cv.GaussianBlur(gray_gradient, (11, 11), 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
    gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    # gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_CLOSE, kernel)

    # gray_gradient = gray_gradient > 100
    # s(gray_gradient)

    # s(gray_gradient)
    canny = gray_gradient
    ## canny = cv.Canny(gray_gradient, 100, 200)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    canny = cv.morphologyEx(canny, cv.MORPH_OPEN, kernel)
    canny = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    canny = cv.morphologyEx(canny, cv.MORPH_ERODE, kernel)
    canny = cv.morphologyEx(canny, cv.MORPH_ERODE, kernel)
    canny = cv.GaussianBlur(canny, (11, 11), 1)

    # orig_canny = canny.copy()
    # canny = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

    # s(cv.vconcat(
    #     [cv.hconcat([cv.cvtColor(gradient, cv.COLOR_BGR2GRAY), pre_threshold_gg]), cv.hconcat([gray_gradient, canny])]))
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35, 35))
    # s(canny)
    return canny


def grade_contours(contours: list[np.ndarray]) -> pd.DataFrame:
    df = []
    # contour_render = np.zeros_like(image)
    # all_contours = np.zeros_like(image)
    smoothing_kernel_size = 101
    for i, contour in enumerate(contours):
        col = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        # cv.drawContours(all_contours, contours, i, col.tolist(), 1)
        area = cv.contourArea(contour)
        if area < 50 * 50:
            # console.print(f"Discarding small contour {i} ({area})")
            continue  # Skip small contours
        # console.print(f"Using contour {i} ({area})")
        new_contour = smooth_contour(contour, kernel_size=smoothing_kernel_size)
        if new_contour is contour:
            console.print(f"Failed to smooth contour {i}, area: {area}")
        contour = new_contour
        area = cv.contourArea(contour)
        if area < 100 * 100:
            # console.print(f"Discarding small smooth contour {i}")
            continue  # Skip small contours
        # cv.drawContours(contour_render, contours, i, col.tolist(), 2)
        # cv.drawContours(contour_render, [contour], -1, col.tolist(), 16)
        # Approximate the contour to a polygon
        approx = cv.approxPolyN(contour, 4)
        # Draw the approximated polygon
        approximation_area = cv.contourArea(approx)
        approx_side_lengths = [
            np.linalg.norm(approx[0][j] - approx[0][(j + 1) % len(approx[0])])
            for j in range(len(approx[0]))
        ]
        side_a_len = approx_side_lengths[0] + approx_side_lengths[2]
        side_b_len = approx_side_lengths[1] + approx_side_lengths[3]
        # Add text with contour index

        # cv.putText(
        #     contour_render,
        #     str(i),
        #     tuple(approx[0][2]),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     3,
        #     col.tolist(),
        #     3,
        # )
        side_a_ratio = (
            approx_side_lengths[0] / approx_side_lengths[2]
            if approx_side_lengths[2] != 0
            else 0
        )
        side_b_ratio = (
            approx_side_lengths[1] / approx_side_lengths[3]
            if approx_side_lengths[3] != 0
            else 0
        )
        aspect_ratio = side_a_len / side_b_len if side_b_len != 0 else 0
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio  # Ensure aspect ratio is >= 1
        ratio_rating = abs((2**0.5) - aspect_ratio)
        side_rating = abs(1 - (side_a_ratio + side_b_ratio) / 2) * 0.8
        area_ratio_rating = abs(1 - area / approximation_area) * 0.2
        area_value_rating = ((max(0, 100_000 - area) / 100_000) ** 0.8) * 1.5
        overall_rating = (
            ratio_rating + side_rating + area_ratio_rating + area_value_rating
        ) / 4
        # cv.drawContours(
        #     contour_render,
        #     [approx],
        #     -1,
        #     lerp_color((255, 0, 0), (0, 255, 255), overall_rating),
        #     4,
        # )
        df.append(
            {
                "index": i,
                "area": area,
                "rect_area": approximation_area,
                "contour_point_count": len(contour),
                # 'side_a': side_a_len,
                # 'side_b': side_b_len,
                # 'side_a_ratio': side_a_ratio,
                # 'side_b_ratio': side_b_ratio,
                "area_ratio_rating": area_ratio_rating,
                "side_rating": side_rating,
                "aspect_ratio": aspect_ratio,
                "area_value_rating": area_value_rating,
                "overall_rating": overall_rating,
            }
        )

    # s(cv.vconcat([cv.cvtColor(canny, cv.COLOR_GRAY2BGR), all_contours, contour_render]))
    df = pd.DataFrame(df)
    if len(df) == 0:
        console.print(f"No contours found")
        df["overall_rating"] = []
    else:
        df["ratio_rating"] = abs((2**0.5) - df["aspect_ratio"])
        df["ratio"] = df["area"] / df["rect_area"]
    df = df.sort_values(by=["overall_rating"], ascending=True)
    console.print(df)
    return df


def render_good_contours(
    canny: np.ndarray,
    smoothing_kernel_size: int = 101,
    df: pd.DataFrame = pd.DataFrame(),
    image: np.ndarray = None,
    rating_threshold: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    good_contours = []
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for i in df[df["overall_rating"] < rating_threshold]["index"]:
        contour_render = np.zeros_like(image)
        mask_render = np.zeros(shape=image.shape[:-1], dtype=np.uint8)
        # console.print(i)
        cv.drawContours(contour_render, contours, i, (0, 255, 0), 20)
        smooth = smooth_contour(contours[i], kernel_size=smoothing_kernel_size)
        cv.drawContours(contour_render, [smooth], 0, (255, 0, 0), 2)
        filled = cv.drawContours(mask_render, [smooth], 0, 255, cv.FILLED)
        good_contours.append(
            (
                contour_render,
                filled,
                cv.bitwise_and(image, image, mask=filled),
                df[df["index"] == i]["overall_rating"].values[0],
            )
        )
    return good_contours
    # if len(good_contours) > 0:
    #     # s(cv.vconcat(good_contours))
    # else:
    #     print(f"No good contours found :c")


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

def process_image(
    smp: DatasetItem, image: np.ndarray, rating_threshold=0.1
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    pre_processed = file_cache(smp.name, "pre_process", lambda: pre_process_image(image))
    contours, hierarchy = cv.findContours(
        pre_processed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )
    contour_data_df = grade_contours(contours)
    return render_good_contours(
        pre_processed,
        smoothing_kernel_size=101,
        df=contour_data_df,
        image=image,
        rating_threshold=rating_threshold,
    )


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


def process_samples(dataset: Dataset, output_dir: Path):
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


def process_sample(smp: DatasetItem, output_dir: Path):
    # try:
    blurry = is_image_blurry3(smp.data)
    is_blurry = blurry < 50
    image_data = smp.data
    processed = process_image(smp, image_data)

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    for i, (contour, mask, mixed, score) in enumerate(processed):
        # mixed = cv.bitwise_and(
        #     image_data, image_data, mask=cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        # )
        area = np.count_nonzero(mask)
        display = np.hstack(
            (image_data, contour, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), mixed)
        )
        # Utils.show_image(display, wait=True)
        is_correct = area > 100_000
        output_path = output_dir / f"{smp.name}_processed_{i}.jpeg"
        cv.imwrite(str(output_path), display)
        mask_path = output_dir / f"{smp.name}_mask_{i}.jpeg"
        cv.imwrite(str(mask_path), mask)
        df_row = {
            "name": smp.name,
            "region_id": i,
            "has_flash": smp.has_flash,
            "has_light": smp.has_light,
            "type": smp.type,
            "regions": len(processed),
            "blur": blurry,
            "is_blurry": is_blurry,
            "area": area,
            "is_correct (estimated)": is_correct,
        }
        yield display if display.shape[0] > 899 else None, df_row
    if len(processed) == 0:
        df_row = {
            "name": smp.name,
            "region_id": -1,
            "has_flash": smp.has_flash,
            "has_light": smp.has_light,
            "type": smp.type,
            "regions": len(processed),
            "blur": blurry,
            "is_blurry": is_blurry,
            "area": 0,
            "is_correct (estimated)": False,
        }
        yield None, df_row

    # log.debug(f"Morph: {morph_non_black_corners / (morph_non_black_processed + 1)}\n" + f"Raw: {raw_non_black_corners / (raw_non_black_processed + 1)}")
    # log.debug(display.shape)

    # Ignore horizontal images.
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

    # Save file in the output directory


# except ValueError:
#     yield None, {}


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
    df = file_cache("full_df", "computed", lambda: compue_df(batch_size, dataset, hide_correct, output_dir, render))
    df = pd.DataFrame(df, columns=[
        "name",
        "region_id",
        "has_flash",
        "has_light",
        "type",
        "regions",
        "blur",
        "is_blurry",
        "area",
        "is_correct (estimated)",
    ])

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

    # Is correct group by (type)
    correct_grouped = (
        df.groupby(["type"])["is_correct (estimated)"]
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
            title="Correct Detections Grouped by Type",
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

    df_group = df.groupby(["name", "has_flash", "has_light"]).agg(
        {
            "is_correct (estimated)": "max",
            "blur": "mean",
            "area": "mean",
            "regions": "mean",
        }
    )

    correct_grouped_total = df_group["is_correct (estimated)"].sum()
    correct_grouped_images = df_group["is_correct (estimated)"].count()
    console.print(Panel(str(df_group), highlight=True, expand=False))
    console.print(
        f"Total correct detections (estimated) grouped: {correct_grouped_total} out of {correct_grouped_images} ({(correct_grouped_total / correct_grouped_images) * 100:.2f}%)"
    )


    if render and not wait_single:
        Utils.show_image(wait=True)


def compue_df(batch_size, dataset, hide_correct, output_dir, render):
    df_rows = []
    proced = process_samples(dataset, output_dir)
    for i, item in enumerate(
            itertools.batched(
                (
                        it
                        # for it in track(proced, total=len(dataset.items), console=console)
                        for it in track(proced, total=140, console=console)
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
        # Utils.show_image(batch, wait=wait_single, offset=0)
    return df_rows


if __name__ == "__main__":
    main()
