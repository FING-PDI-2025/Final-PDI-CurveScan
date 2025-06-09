import cv2
import numpy as np
from pathlib import Path
import logging
from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console

from utils import ImageProcessingIntermediateStepReporter, Img

A4_DIM = (1240, 1754)  # A4 dimensions in px (width, height)

# Set up rich logger and console
console = Console()
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console)]
)
log = logging.getLogger("rich")


def sort_corners(corners):
    """Sort the corners to get [top-left, top-right, bottom-right, bottom-left]"""
    # Sort by Y coordinate (top to bottom)
    corners = sorted(corners, key=lambda x: x[1])

    # Get top two and bottom two corners
    top_corners = sorted(corners[:2], key=lambda x: x[0])
    bottom_corners = sorted(corners[2:], key=lambda x: x[0])

    # Return in the order [top-left, top-right, bottom-right, bottom-left]
    return [top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]]


def find_document_corners(mask_path):
    """Find the corners of the document in the binary mask"""
    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        log.error(f"Failed to read mask: {mask_path}")
        return None

    # Get image dimensions
    img_height, img_width = mask.shape
    total_img_area = img_height * img_width

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return None
    if not contours:
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate area coverage percentage
    contour_area = cv2.contourArea(largest_contour)
    coverage_percentage = (contour_area / total_img_area) * 100

    # Check if document is too small (little coverage of the image)
    if coverage_percentage < 10:
        log.error(
            f"Document in {mask_path} covers only {coverage_percentage:.1f}% of the image. Document may be too far from camera."
        )
        return None

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If the polygon has 4 corners, use those, otherwise find the minimum area rectangle
    if len(approx) == 4:
        corners = [corner[0] for corner in approx]
    elif len(approx) > 4:
        # If more than 4 corners, use a minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        corners = [tuple(map(int, corner)) for corner in box]
    else:
        log.error(
            f"Could not find 4 corners in the contour approximation for {mask_path}. Found: {len(approx)}"
        )
        return None

    # Sort corners to get [top-left, top-right, bottom-right, bottom-left]
    return sort_corners(corners)


def get_output_dimensions(corners, max_dimension=A4_DIM):
    """Calculate output dimensions based on detected document shape, maximizing size within limits"""
    # Calculate width and height of the detected document
    x_coords = [corner[0] for corner in corners]
    y_coords = [corner[1] for corner in corners]
    doc_width = max(x_coords) - min(x_coords)
    doc_height = max(y_coords) - min(y_coords)

    # Calculate aspect ratio of the detected document
    doc_ratio = doc_height / doc_width if doc_width > 0 else 1.0

    # Default to A4 aspect ratio if near A4 dimensions
    if doc_ratio > 1.05 and doc_ratio < 1.77:
        doc_ratio = 1.41  # Vertical A4 aspect ratio
    if doc_ratio > 0.55 and doc_ratio < 0.85:
        doc_ratio = 0.7  # Horizontal A4 aspect ratio

    # Get maximum allowed dimensions
    max_width, max_height = max_dimension

    # Determine the best fit while maximizing dimensions
    if doc_ratio > 1:  # Taller than wide (portrait)
        if max_height / doc_ratio <= max_width:
            # Height limited
            height = max_height
            width = int(height / doc_ratio)
        else:
            # Width limited
            width = max_width
            height = int(width * doc_ratio)
    else:  # Wider than tall (landscape)
        if max_width * doc_ratio <= max_height:
            # Width limited
            width = max_width
            height = int(width * doc_ratio)
        else:
            # Height limited
            height = max_height
            width = int(height / doc_ratio)

    # Ensure minimum dimensions and avoid zero values
    width = max(100, width)
    height = max(100, height)

    log.info(f"Original document ratio: {doc_ratio:.2f}, output size: {width}x{height}")

    return (width, height)


def apply_perspective_transform(image_path, corners, output_size=(1080, 1080)):
    """Apply perspective transform to get a square document"""
    # Read the image
    image = cv2.imread(str(image_path))

    if image is None:
        log.error(f"Failed to read image: {image_path}")
        return None

    # Validate corners
    if len(corners) != 4:
        log.error(
            f"Perspective transform requires exactly 4 corners, got {len(corners)}"
        )
        return None

    # Define the destination points for the perspective transform
    # This will be a square
    dst_points = np.array(
        [
            [0, 0],  # top-left
            [output_size[0], 0],  # top-right
            [output_size[0], output_size[1]],  # bottom-right
            [0, output_size[1]],  # bottom-left
        ],
        dtype=np.float32,
    )

    # Convert corners to numpy array
    src_points = np.array(corners, dtype=np.float32)

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, output_size)

    return warped


def enhance_image(image: Img, reporter: ImageProcessingIntermediateStepReporter) -> tuple[Path, Img, list[str]]:
    """Enhance image based on analysis results"""
    enhanced = image.copy()
    enhancements_applied = []

    # ------ Apply white balance as first step ------
    # Simple white balance using gray world assumption
    b, g, r = cv2.split(enhanced)

    # Calculate average of each channel
    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)

    # Calculate the gray average
    gray_avg = (b_avg + g_avg + r_avg) / 3

    # Scale the channels to balance them
    b_scaled = np.clip((b * (gray_avg / b_avg)), 0, 255).astype(np.uint8)
    g_scaled = np.clip((g * (gray_avg / g_avg)), 0, 255).astype(np.uint8)
    r_scaled = np.clip((r * (gray_avg / r_avg)), 0, 255).astype(np.uint8)

    # Merge the balanced channels
    enhanced = cv2.merge([b_scaled, g_scaled, r_scaled])
    reporter.output("white_balanced", enhanced)
    enhancements_applied.append("white balanced")

    # ------ Apply sharpening filter (to enhance details) ------
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    reporter.output("sharpened", enhanced)
    enhancements_applied.append("sharpened")

    # ------ Apply bilateral filter to reduce noise while preserving edges ------
    # This is especially good for text documents
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    reporter.output("bilateral_filtered", enhanced)
    enhancements_applied.append("noise reduced")

    # ------ Apply morphological erode to improve text clarity ------
    # This is especially useful for removing small noise while preserving text shapes
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # For color images, apply to each channel separately
    b, g, r = cv2.split(enhanced)
    b = cv2.morphologyEx(b, cv2.MORPH_ERODE, kernel)
    g = cv2.morphologyEx(g, cv2.MORPH_ERODE, kernel)
    r = cv2.morphologyEx(r, cv2.MORPH_ERODE, kernel)
    enhanced = cv2.merge([b, g, r])
    reporter.output("morphological_erode", enhanced)
    enhancements_applied.append("morphological opening applied")

    # ------ Apply Sobel edge detection ------
    # Convert to grayscale for edge detection
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Use Sobel operator to find edges
    sobel_x = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Convert to 8-bit
    sobel_abs = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    reporter.output("sobel_edges", sobel_abs)

    # Combine Sobel edges with the enhanced image to improve detail sharpness
    # For color images, we need to convert Sobel to 3-channel
    sobel_colored = cv2.cvtColor(sobel_abs, cv2.COLOR_GRAY2BGR)

    # Use the Sobel edges as an overlay with alpha blending
    alpha = 0.4  # Adjust strength of edge enhancement
    enhanced = cv2.addWeighted(enhanced, 1.0, sobel_colored, alpha, 0)
    reporter.output("sobel_colored", enhanced)
    enhancements_applied.append("edge enhancement applied")

    # Save partial enhanced image

    # ------ Apply morphological opening to improve text clarity ------
    # This is especially useful for removing small noise while preserving text shapes
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # For color images, apply to each channel separately
    b, g, r = cv2.split(enhanced)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    r = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel)
    enhanced = cv2.merge([b, g, r])
    output_path = reporter.output("morphological_opening", enhanced, kind="output")
    enhancements_applied.append("morphological opening applied")

    return output_path, enhanced, enhancements_applied


def process_document(original_path, mask_path, output_path=None, max_dimension=A4_DIM):
    """Process a document image: find corners and apply perspective transform"""

    output_dir = Path(output_path).parent if output_path else None
    # If output_path is not provided, create one based on the original path
    if output_path is None:
        output_dir = Path(original_path).parent
        output_filename = f"{Path(original_path).stem}_transformed.jpg"
        output_path = output_dir / output_filename

    # Find document corners from mask
    corners = find_document_corners(mask_path)

    if corners is None:
        log.error(f"Could not find document corners in {mask_path}")
        return False

    # Calculate output dimensions
    output_size = get_output_dimensions(corners, max_dimension)

    # Apply perspective transform
    transformed = apply_perspective_transform(original_path, corners, output_size)
    final_image = transformed

    if transformed is None:
        return False

    # Enhance image quality
    _, enhanced, enhancements = enhance_image(
        transformed, str(output_dir / f"{Path(output_path).stem}_zap.jpg")
    )
    final_image = enhanced

    if len(enhancements) > 0:
        log.info(f"Applied enhancements: {', '.join(enhancements)}")

        # Create an output path for the enhanced image
        transformed_path = output_dir / f"{Path(output_path).stem}_partial.jpg"

        # Save previous result for comparison
        cv2.imwrite(str(transformed_path), transformed)
        log.info(f"Intermediate image saved to {transformed_path}")

    # Save the transformed and enhanced image
    cv2.imwrite(str(output_path), final_image)
    log.info(f"Transformed document saved to {output_path}")

    # Save same image to results directory
    results_path = (
        output_dir.parent.parent / "results" / f"{Path(output_path).stem}_result.jpg"
    )
    cv2.imwrite(str(results_path), final_image)
    log.info(f"Result image saved to {results_path}")

    # Save same original image to results directory
    original_results_path = (
        output_dir.parent.parent
        / "results"
        / f"{Path(original_path).stem}_original.jpg"
    )
    cv2.imwrite(str(original_results_path), cv2.imread(str(original_path)))
    log.info(f"Original image saved to {original_results_path}")

    # Return success status
    return True


def batch_process(
    input_dir, output_dir=None, pattern="*_mask_*.jpeg", max_dimension=A4_DIM
):
    """Process all documents in a directory based on their mask files"""
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = input_path / "transformed"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = output_dir.parent.parent / "results"
    # Create results directory if it doesn't exist
    result_dir.mkdir(parents=True, exist_ok=True)

    # Find all mask files
    mask_files = list(input_path.glob(pattern))
    log.info(f"Found {len(mask_files)} mask files to process")

    success_count = 0
    for mask_file in track(mask_files, description="Processing documents", console=console):
        # Derive original image path
        # Assuming format: name_flash/noflash_light/nolight_mask.png
        mask_extra = mask_file.stem[mask_file.stem.index("_mask"):]
        stem = mask_file.stem.replace(mask_extra, "")
        datasets_dir = input_path.parent / "datasets"
        original_file = None

        # Check all subdirectories of datasets
        if datasets_dir.exists():
            # Search in all subdirectories
            rglob = list(datasets_dir.rglob(f"{stem}*"))
            if len(rglob) > 1:
                log.warning(
                    f"Multiple files found for {stem} in datasets: {rglob}. Using the first one."
                )
            original_file = rglob[0]

        # If not found, try with the original expected path
        if not original_file:
            original_file = datasets_dir / "caratulas" / f"{stem}.jpeg"

        # If we still can't find the original, skip
        if not original_file.exists():
            log.warning(f"Could not find original image for mask: {mask_file}")
            continue

        # Define output path
        output_file = output_dir / f"{mask_file.stem}_transformed.jpg"

        ok = process_document(original_file, mask_file, output_file, max_dimension)
        # Process the document
        if ok:
            success_count += 1

    log.info(
        f"Successfully processed {success_count} out of {len(mask_files)} documents"
    )


def main():
    batch_process("../outputs")


if __name__ == "__main__":
    main()
