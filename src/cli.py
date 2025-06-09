import numpy as np
from pathlib import Path
from typing import Sequence
import cv2 as cv
import pandas as pd
import logging
from rich.logging import RichHandler
from rich.console import Console

from dataset_item import DatasetItem
import tps_transform_for_cli
from utils import ImageProcessingIntermediateStepReporter, Img, adjust_gamma, smooth_contour
from typer import Typer

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")
pd.set_option("display.max_rows", 500)
app = Typer()
console = Console()
debug = False


def pre_process_image(image: Img, reporter: ImageProcessingIntermediateStepReporter) -> Img:
    with reporter.with_indent("pre_process") as r:
        r.output("original", image)
        blurred = cv.medianBlur(image, 5)
        r.output("blurred", blurred)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        gradient = cv.morphologyEx(blurred, cv.MORPH_GRADIENT, kernel)
        r.output("gradient", gradient)
        gray_gradient = cv.cvtColor(gradient, cv.COLOR_BGR2GRAY)
        gray_gradient_eq = cv.equalizeHist(gray_gradient)
        gray_gradient = cv.addWeighted(gray_gradient_eq, 0.15, gray_gradient, 6, 0)
        gray_gradient = adjust_gamma(gray_gradient, 0.60)
        r.output("gray_gradient", gray_gradient)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
        r.output("dilated", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_OPEN, kernel)
        r.output("opened", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_CLOSE, kernel)
        r.output("closed", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_ERODE, kernel)
        r.output("eroded", gray_gradient)
        gray_gradient = cv.GaussianBlur(gray_gradient, (5, 5), 1)
        r.output("blurred", gray_gradient)

        _, gray_gradient = cv.threshold(gray_gradient, -1, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
        r.output("thresholded", gray_gradient)
        gray_gradient = cv.GaussianBlur(gray_gradient, (11, 11), 1)
        r.output("blurred", gray_gradient)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
        r.output("dilated", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
        r.output("dilated", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_DILATE, kernel)
        r.output("dilated", gray_gradient)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_OPEN, kernel)
        r.output("opened", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_CLOSE, kernel)
        r.output("closed", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_ERODE, kernel)
        r.output("eroded", gray_gradient)
        gray_gradient = cv.morphologyEx(gray_gradient, cv.MORPH_ERODE, kernel)
        r.output("eroded", gray_gradient)
        gray_gradient = cv.GaussianBlur(gray_gradient, (11, 11), 1)
        r.output("blurred", gray_gradient)

        return gray_gradient


def grade_contours(
    image_shape, contours: Sequence[Img], smoothing_kernel_size, reporter: ImageProcessingIntermediateStepReporter
) -> pd.DataFrame:
    with reporter.with_indent("contour_grading") as r:
        df = []
        all_detected_contours = np.zeros(shape=image_shape, dtype=np.uint8)
        smoothed_contours = np.zeros(shape=image_shape, dtype=np.uint8)

        for i, contour in enumerate(contours):
            col = np.random.randint(0, 255, size=3, dtype=np.uint8)
            cv.drawContours(all_detected_contours, contours, i, col.tolist(), 1)
            area = cv.contourArea(contour)
            if area < 50 * 50:
                continue
            new_contour = smooth_contour(contour, kernel_size=smoothing_kernel_size)
            if new_contour is contour:
                console.print(f"Failed to smooth contour {i}, area: {area}")
            cv.drawContours(smoothed_contours, [contour], 0, col.tolist(), 1)
            contour = new_contour
            cv.drawContours(smoothed_contours, [contour], 0, col.tolist(), 8)
            area = cv.contourArea(contour)
            if area < 100 * 100:
                continue
            approx = cv.approxPolyN(contour, 4)

            approximation_area = cv.contourArea(approx)
            approx_side_lengths = [
                np.linalg.norm(approx[0][j] - approx[0][(j + 1) % len(approx[0])]) for j in range(len(approx[0]))
            ]
            side_a_len = approx_side_lengths[0] + approx_side_lengths[2]
            side_b_len = approx_side_lengths[1] + approx_side_lengths[3]

            side_a_ratio = approx_side_lengths[0] / approx_side_lengths[2] if approx_side_lengths[2] != 0 else 0
            side_b_ratio = approx_side_lengths[1] / approx_side_lengths[3] if approx_side_lengths[3] != 0 else 0
            aspect_ratio = side_a_len / side_b_len if side_b_len != 0 else 0
            if aspect_ratio < 1:
                aspect_ratio = 1 / aspect_ratio
            ratio_rating = abs((2**0.5) - aspect_ratio)
            side_rating = abs(1 - (side_a_ratio + side_b_ratio) / 2) * 0.8
            area_ratio_rating = abs(1 - area / approximation_area) * 0.2
            area_value_rating = ((max(0, 100_000 - area) / 100_000) ** 0.8) * 1.5
            overall_rating = (ratio_rating + side_rating + area_ratio_rating + area_value_rating) / 4

            df.append(
                {
                    "index": i,
                    "area": area,
                    "rect_area": approximation_area,
                    "contour_point_count": len(contour),
                    "area_ratio_rating": area_ratio_rating,
                    "side_rating": side_rating,
                    "aspect_ratio": aspect_ratio,
                    "area_value_rating": area_value_rating,
                    "overall_rating": overall_rating,
                }
            )
        reporter.output("all_detected_contours", all_detected_contours)
        reporter.output("smoothed_contours", smoothed_contours)
        df = pd.DataFrame(df)
        if len(df) == 0:
            console.print(f"No contours found")
            df["overall_rating"] = []
        else:
            df["ratio_rating"] = abs((2**0.5) - df["aspect_ratio"])
            df["ratio"] = df["area"] / df["rect_area"]
        df = df.sort_values(by=["overall_rating"], ascending=True)
        if debug:
            console.print(df)
        return df


def render_good_contours(
    canny: np.ndarray,
    smoothing_kernel_size: int,
    df: pd.DataFrame,
    image: np.ndarray,
    rating_threshold: float,
    reporter: ImageProcessingIntermediateStepReporter,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    with reporter.with_indent("render_good_contours") as r:
        good_contours = []
        contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for i in df[df["overall_rating"] < rating_threshold]["index"]:
            contour_render = np.zeros_like(image)
            mask_render = np.zeros(shape=image.shape[:-1], dtype=np.uint8)

            cv.drawContours(contour_render, contours, i, (0, 255, 0), 20)
            smooth = smooth_contour(contours[i], kernel_size=smoothing_kernel_size)
            cv.drawContours(contour_render, [smooth], 0, (255, 0, 0), 2)
            filled_mask = cv.drawContours(mask_render, [smooth], 0, 255, cv.FILLED)
            masked_image = cv.bitwise_and(image, image, mask=filled_mask)
            reporter.output(f"good_contour_{i:04d}", contour_render)
            reporter.output(f"good_mask_{i:04d}", filled_mask)
            reporter.output(f"masked_image_{i:04d}", masked_image)
            good_contours.append(
                (
                    contour_render,
                    filled_mask,
                    masked_image,
                    df[df["index"] == i]["overall_rating"].values[0],
                )
            )
        return good_contours


def process_image(
    image: np.ndarray,
    rating_threshold,
    smoothing_kernel_size: int,
    reporter: ImageProcessingIntermediateStepReporter,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    pre_processed = pre_process_image(image, reporter)
    contours, hierarchy = cv.findContours(pre_processed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour_data_df = grade_contours(image.shape, contours, smoothing_kernel_size, reporter)
    return render_good_contours(
        pre_processed,
        smoothing_kernel_size=smoothing_kernel_size,
        df=contour_data_df,
        image=image,
        rating_threshold=rating_threshold,
        reporter=reporter,
    )


def process_sample(
    smp: DatasetItem,
    reporter: ImageProcessingIntermediateStepReporter,
):

    image_data = smp.data
    processed = process_image(image_data, 0.1, 101, reporter)

    for i, (contour, mask, mixed, score) in enumerate(processed):
        area = np.count_nonzero(mask)
        display = np.hstack((image_data, contour, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), mixed))
        output_path = reporter.output("display", display, kind="display")
        mask_path = reporter.output("mask", mask, kind="mask")
        yield {
            "name": smp.name,
            "region_id": i,
            "has_flash": smp.has_flash,
            "has_light": smp.has_light,
            "type": smp.type,
            "regions": len(processed),
            "area": area,
            "mask_path": str(mask_path),
            "output_path": str(output_path),
        }


@app.command()
def transform_image(input_path: Path, output_dir: Path | None = None) -> None:
    assert input_path.exists()
    if output_dir is None:
        output_dir = Path("output") / input_path.stem
    if output_dir.exists():
        raise FileExistsError(output_dir.absolute())
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = ImageProcessingIntermediateStepReporter(output_dir)
    image = cv.imread(str(input_path.absolute()), cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image {input_path.absolute()} not found")
    for df_row in process_sample(DatasetItem(path=input_path, _base_dataset=None), reporter):
        mask_path = Path(df_row["mask_path"])
        try:
            tps_transform_for_cli.process_image(
                input_path,
                mask_path,
                100,
                smoothing=3,
                sampling_method='spline',
                reporter=reporter,
            )
        except ValueError as e:
            console.print(f"Failed to transform image {input_path}: {e}")


if __name__ == "__main__":
    app()
