import cv2
import numpy as np
from pathlib import Path
import logging
from rich.logging import RichHandler
from rich.progress import track
from rich.console import Console

A4_DIM = (1240, 1754)  # A4 dimensions in px (width, height)

# Set up rich logger and console
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")
console = Console()


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


def analyze_image(image):
    """Analyze image for quality issues and return analysis results"""
    results = {}

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate brightness
    brightness = np.mean(gray)
    results["brightness"] = brightness

    # Calculate contrast
    contrast = np.std(gray)
    results["contrast"] = contrast

    # Calculate blurriness using Laplacian variance method
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    results["sharpness"] = laplacian_var

    # Calculate noise level using median absolute deviation
    median = np.median(gray)
    mad = np.median(np.abs(gray - median))
    results["noise"] = mad

    # Determine image quality based on thresholds
    results["is_too_bright"] = brightness > 180
    results["is_too_dark"] = brightness < 50
    results["has_low_contrast"] = contrast < 40
    results["is_blurry"] = laplacian_var < 100
    results["is_noisy"] = mad > 10

    return results


def enhance_image(image, analysis, output_path):
    """Enhance image based on analysis results"""
    enhanced = image.copy()
    enhancements_applied = []

    # Apply white balance as first step
    if len(enhanced.shape) == 3:
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

        enhancements_applied.append("white balanced")
    else:
        # For grayscale images - no white balance needed
        pass

    # Check if image is blurry
    if True or analysis["is_blurry"]:
        # Apply sharpening filter
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhancements_applied.append("sharpened")

    # Check if image is noisy
    if True or analysis["is_noisy"]:
        # Apply bilateral filter to reduce noise while preserving edges
        # This is especially good for text documents
        if len(enhanced.shape) == 3:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        else:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        enhancements_applied.append("noise reduced")

    # Generate Laplacian of the image for edge detection visualization
    if len(enhanced.shape) == 3:
        # Convert to grayscale for Laplacian
        gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        gray_enhanced = enhanced.copy()
        # Apply adaptive thresholding to Laplacian to remove noise

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray_enhanced, cv2.CV_64F)

    # Scale to visible range
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # First normalize Laplacian to 8-bit range for thresholding
    laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Apply simple threshold to get binary edges
    _, thresh_value = cv2.threshold(
        laplacian_norm, 100, 255, cv2.THRESH_BINARY  # Fixed threshold value
    )

    # Combine original Laplacian with thresholded version to keep edge strength
    # but remove noise in flat areas
    laplacian_abs = cv2.bitwise_and(laplacian_abs, thresh_value)

    # Create a path for the Laplacian image
    cv2.imwrite(str(output_path), laplacian_abs)
    log.info(f"Laplacian image saved to {output_path}")

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Convert to 8-bit
    sobel_abs = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Create a path for the Sobel image
    sobel_path = str(output_path).replace(".jpg", "_sobel.jpg")
    cv2.imwrite(sobel_path, sobel_abs)
    log.info(f"Sobel edge image saved to {sobel_path}")

    # Combine original Laplacian with thresholded version to keep edge strength
    # but remove noise in flat areas
    laplacian_abs = cv2.bitwise_and(laplacian_abs, thresh_value)

    # Create a path for the Laplacian image
    cv2.imwrite(str(output_path), laplacian_abs)
    log.info(f"Laplacian image saved to {output_path}")

    # Combine Sobel edges with the enhanced image to improve detail sharpness
    if len(enhanced.shape) == 3:
        # For color images, we need to convert Sobel to 3-channel
        sobel_colored = cv2.cvtColor(sobel_abs, cv2.COLOR_GRAY2BGR)

        # Use the Sobel edges as an overlay with alpha blending
        alpha = 0.3  # Adjust strength of edge enhancement
        enhanced = cv2.addWeighted(enhanced, 1.0, sobel_colored, alpha, 0)
        enhancements_applied.append("edge enhancement applied")
    else:
        # For grayscale images
        alpha = 0.3
        enhanced = cv2.addWeighted(enhanced, 1.0, sobel_abs, alpha, 0)
        enhancements_applied.append("edge enhancement applied")

    # Apply morphological opening to improve text clarity
    # This is especially useful for removing small noise while preserving text shapes
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if len(enhanced.shape) == 3:
        # For color images, apply to each channel separately
        b, g, r = cv2.split(enhanced)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
        r = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel)
        enhanced = cv2.merge([b, g, r])
    else:
        # For grayscale images
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

    enhancements_applied.append("morphological opening applied")

    return enhanced, enhancements_applied


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

    if transformed is None:
        return False

    # Analyze the transformed image for quality issues
    analysis = analyze_image(transformed)

    # Log the image analysis
    quality_issues = []
    if analysis["is_too_bright"]:
        quality_issues.append("too bright")
    if analysis["is_too_dark"]:
        quality_issues.append("too dark")
    if analysis["has_low_contrast"]:
        quality_issues.append("low contrast")
    if analysis["is_blurry"]:
        quality_issues.append("blurry")
    if analysis["is_noisy"]:
        quality_issues.append("noisy")

    final_image = transformed

    if quality_issues:
        log.info(f"Image quality issues detected: {', '.join(quality_issues)}")

        # Enhance image based on analysis
        enhanced, enhancements = enhance_image(
            transformed, analysis, str(output_dir / f"{Path(output_path).stem}_zap.jpg")
        )
        final_image = enhanced

        if enhancements:
            log.info(f"Applied enhancements: {', '.join(enhancements)}")

            # Create an output path for the enhanced image
            transformed_path = output_dir / f"{Path(output_path).stem}_partial.jpg"

            # Save previous result for comparison
            cv2.imwrite(str(transformed_path), transformed)
            log.info(f"Intermediate image saved to {transformed_path}")

    # Save the transformed and enhanced image
    cv2.imwrite(str(output_path), final_image)
    log.info(f"Transformed document saved to {output_path}")

    return True


def batch_process(
    input_dir, output_dir=None, pattern="*_mask.png", max_dimension=A4_DIM
):
    """Process all documents in a directory based on their mask files"""
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = input_path / "transformed"
    else:
        output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all mask files
    mask_files = list(input_path.glob(pattern))
    log.info(f"Found {len(mask_files)} mask files to process")

    success_count = 0
    for mask_file in track(mask_files, description="Processing documents"):
        # Derive original image path
        # Assuming format: name_flash/noflash_light/nolight_mask.png
        stem = mask_file.stem.replace("_mask", "")
        datasets_dir = input_path.parent / "datasets" / "caratulas"
        original_file = datasets_dir / f"{stem}.jpeg"
        if datasets_dir.exists():
            for ext in [".jpeg", ".jpg", ".png"]:
                test_path = datasets_dir / f"{stem}{ext}"
                if test_path.exists():
                    original_file = test_path
                    break

        # If we still can't find the original, skip
        if not original_file.exists():
            log.warning(f"Could not find original image for mask: {mask_file}")
            continue

        # Define output path
        output_file = output_dir / f"{stem}_transformed.jpg"

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
