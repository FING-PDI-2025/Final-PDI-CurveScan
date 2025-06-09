# Apply TPS transform to the mask
from typing import Literal

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splprep, splev, Rbf
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

from utils import Contour, ImageProcessingIntermediateStepReporter, Img


def extract_main_contour(mask: Img) -> Contour:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour


def spline_fit_and_sample(contour: Contour, num_points: int = 50, input_resolution: int = 100, smoothing: int = 3):
    # Downsample contour points carefully
    sampled_contour = contour[::input_resolution]

    # Remove duplicate points
    _, idx = np.unique(sampled_contour, axis=0, return_index=True)
    unique_points = sampled_contour[np.sort(idx)]
    if len(unique_points) < 4:
        raise ValueError("Not enough unique points for spline fitting. Adjust input_resolution or smoothing.")

    x, y = unique_points[:, 0], unique_points[:, 1]

    print(len(x), len(y))
    # Fit spline ensuring periodicity (closed contour)
    tck, u = splprep([x, y], s=smoothing, per=False)

    # Uniformly sample points based on spline parameterization
    unew = np.linspace(0, 1.0, num_points)
    out = splev(unew, tck)

    return np.column_stack(out)


def get_sorted_quadrilateral_corners(contour: Contour) -> Contour:
    """
    Detects 4-corner quadrilateral from a contour and returns corners
    in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) != 4:
        raise ValueError("Contour does not approximate a quadrilateral.")

    # Ensure it's a float32 array for further operations
    corners = approx.astype(np.float32)

    # Compute pairwise distances
    dists = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]

    # Create 2 pairs: [0-1, 2-3] and [1-2, 3-0]
    pair_01_23 = (corners[0], corners[1], corners[2], corners[3])  # edge 01 and 23
    pair_12_30 = (corners[1], corners[2], corners[3], corners[0])  # edge 12 and 30

    length_01_23 = np.linalg.norm(pair_01_23[1] - pair_01_23[0]) + np.linalg.norm(pair_01_23[2] - pair_01_23[3])
    length_12_30 = np.linalg.norm(pair_12_30[1] - pair_12_30[0]) + np.linalg.norm(pair_12_30[2] - pair_12_30[3])

    if length_01_23 < length_12_30:
        # Short edges: top and bottom
        top = np.array([pair_01_23[0], pair_01_23[1]])
        bottom = np.array([pair_01_23[3], pair_01_23[2]])
    else:
        # Short edges: left and right
        left = np.array([pair_12_30[0], pair_12_30[1]])
        right = np.array([pair_12_30[3], pair_12_30[2]])

        # Assign top and bottom based on y
        if np.mean(left[:, 1]) < np.mean(right[:, 1]):
            top = left
            bottom = right
        else:
            top = right
            bottom = left

    # Order corners: TL, TR, BR, BL
    if top[0][0] < top[1][0]:
        tl, tr = top[0], top[1]
    else:
        tl, tr = top[1], top[0]

    if bottom[0][0] < bottom[1][0]:
        bl, br = bottom[0], bottom[1]
    else:
        bl, br = bottom[1], bottom[0]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return ordered


def is_contour_clockwise(contour: Contour) -> bool:
    """
    Uses the signed area (Shoelace formula) to check if a contour is clockwise.
    Positive area → clockwise; Negative → counterclockwise.
    """
    x = contour[:, 0]
    y = contour[:, 1]
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0


def enforce_clockwise_contour(contour: Contour) -> Contour:
    """
    Reverses contour points if they are counterclockwise.
    """
    if not is_contour_clockwise(contour):
        return contour[::-1]
    return contour


def get_nearest_index_in_contour(contour: Contour, point: np.ndarray) -> int:
    """
    Finds the index of the point in the contour that is closest to the given point.
    """
    dists = np.linalg.norm(contour - point, axis=1)
    return np.argmin(dists)


def split_contour_edges(contour: Contour, corner_indices: list[int]) -> list[Contour]:
    """
    Splits a closed contour into 4 edge segments using corner indices.
    For each edge, chooses the shorter direction (forward or backward).
    """
    num_points = len(contour)
    edges = []

    for i in range(4):
        start_idx = corner_indices[i]
        end_idx = corner_indices[i + 1]

        # Forward path
        if start_idx <= end_idx:
            forward = contour[start_idx : end_idx + 1]
        else:
            forward = np.vstack([contour[start_idx:], contour[: end_idx + 1]])

        # Backward path
        if end_idx <= start_idx:
            backward = contour[end_idx : start_idx + 1][::-1]
        else:
            backward = np.vstack([contour[end_idx:], contour[: start_idx + 1]])[::-1]

        # Choose shorter path
        if len(forward) <= len(backward):
            edges.append(forward)
        else:
            edges.append(backward)

    return edges


def sample_contour_edges_with_splines(edges: list[Contour], points_per_side: int, smoothing: int) -> Contour:
    all_samples = []
    for edge in edges:
        # Ensure edges go forward
        if len(edge) < 2:
            continue
        # Flip if needed
        if np.linalg.norm(edge[0] - edge[-1]) > np.linalg.norm(edge[-1] - edge[0]):
            edge = edge[::-1]

        samples = spline_fit_and_sample(edge, num_points=points_per_side, smoothing=smoothing)
        all_samples.append(samples)
    return np.vstack(all_samples)


def sample_contour_edges(
    edges: list[Contour], points_per_side: int, smoothing: int = 3, method: Literal["spline", "contour_arc"] = "spline"
) -> Contour:
    all_samples = []

    for edge in edges:
        if len(edge) < 2:
            continue

        match method:
            case "spline":
                samples = spline_fit_and_sample(edge, num_points=points_per_side, smoothing=smoothing)
            case "contour_arc":
                # Uniform arc-length sampling
                dists = np.linalg.norm(np.diff(edge, axis=0), axis=1)
                cumlen = np.hstack(([0], np.cumsum(dists)))
                total_len = cumlen[-1]
                target_lengths = np.linspace(0, total_len, points_per_side)

                sampled = []
                for t in target_lengths:
                    idx = np.searchsorted(cumlen, t)
                    if idx == 0:
                        sampled.append(edge[0])
                    elif idx >= len(edge):
                        sampled.append(edge[-1])
                    else:
                        alpha = (t - cumlen[idx - 1]) / (cumlen[idx] - cumlen[idx - 1])
                        pt = (1 - alpha) * edge[idx - 1] + alpha * edge[idx]
                        sampled.append(pt)
                samples = np.vstack(sampled)

            case _:
                raise ValueError(f"Unknown sampling method: {method}")

        all_samples.append(samples)

    return np.vstack(all_samples)


def generate_destination_rectangle(corners: Contour, points_per_side: int) -> tuple[Contour, tuple[int, int]]:
    width = np.mean(
        [
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[3]),
        ]
    )
    height = np.mean(
        [
            np.linalg.norm(corners[3] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
        ]
    )

    dst_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    dst_edges = []
    for i in range(4):
        start = dst_corners[i]
        end = dst_corners[(i + 1) % 4]
        edge_points = np.linspace(start, end, points_per_side, endpoint=False)
        dst_edges.append(edge_points)
    return np.vstack(dst_edges), (int(width), int(height))


def plot_split_edges_colored(
    edges: list[Contour], corners: Contour, background: Img, reporter: ImageProcessingIntermediateStepReporter
) -> None:
    """
    Plot split contour edges in different colors with labels (Top, Right, Bottom, Left).

    Parameters:
    - edges: list of 4 (N_i, 2) numpy arrays (from split_contour_edges)
    - title: plot title
    - background: optional background image (e.g., the mask)
    """
    assert len(edges) == 4, f"Expected 4 contour edge segments, got {len(edges)}\n{edges}"
    edge_colors = ["red", "green", "blue", "orange"]
    edge_labels = ["Top", "Right", "Bottom", "Left"]
    corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

    plt.figure(figsize=(8, 8))
    if background is not None:
        plt.imshow(background, cmap="gray")

    for i, edge in enumerate(edges):
        edge = np.array(edge)
        plt.plot(
            edge[:, 0],
            edge[:, 1],
            marker="o",
            markersize=2,
            linestyle="-",
            linewidth=1.0,
            color=edge_colors[i],
            label=f"{edge_labels[i]} (Edge {i})",
        )

        # Label each edge in the middle
        mid_idx = len(edge) // 2
        mid_pt = edge[mid_idx]
        plt.text(
            mid_pt[0],
            mid_pt[1],
            edge_labels[i],
            fontsize=10,
            color=edge_colors[i],
            weight="bold",
        )
    # Plot corners
    for i, corner in enumerate(corners):
        plt.scatter(corner[0], corner[1], color="lime", s=20, label=f"{corner_labels[i]}")
        plt.text(corner[0], corner[1], corner_labels[i], fontsize=10, color="lime")

    plt.title("Split Contour Edges with Corners")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    reporter.output_mpl("split_contour_edges", plt.gcf())


def prepare_tps_points_from_mask(
    mask: Img,
    points_per_side: int,
    smoothing: int,
    sampling_method: Literal["spline", "contour_arc"],
    reporter: ImageProcessingIntermediateStepReporter,
) -> tuple[Contour, Contour]:
    contour = extract_main_contour(mask)
    contour = enforce_clockwise_contour(contour)
    corners = get_sorted_quadrilateral_corners(contour)
    nearest_index_per_corner = [get_nearest_index_in_contour(contour, corner) for corner in corners]
    nearest_index_per_corner = [
        *nearest_index_per_corner,
        nearest_index_per_corner[0],
    ]  # Close loop

    edges = split_contour_edges(contour, nearest_index_per_corner)
    plot_split_edges_colored(edges, corners, background=mask, reporter=reporter)
    src_points = sample_contour_edges(edges, points_per_side, smoothing=smoothing, method=sampling_method)
    dst_points, rect_dims = generate_destination_rectangle(corners, points_per_side)
    return src_points, dst_points


def apply_tps_transform_skimage(
    image: Img, src_points: Contour, dst_points: Contour, output_shape: tuple[int, ...] = None
) -> Img:
    assert image is not None and image.ndim == 3, "Expected HWC image"
    assert src_points.shape == dst_points.shape and len(src_points) >= 3

    # Convert image to float in [0, 1]
    image = image.astype(np.float32) / 255.0

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_points, src_points)  # dst → src mapping

    if output_shape is None:
        output_shape = image.shape[:2]

    # Warp with linear interpolation, preserving range and avoiding black edges
    warped = warp(
        image,
        tform,
        output_shape=output_shape,
        mode="edge",
        order=1,
        preserve_range=True,
    )

    # Convert back to uint8
    return (warped * 255).astype(np.uint8)


def process_image(
    original_image_path: Path,
    mask_image_path: Path,
    points_per_side: int,
    smoothing: int,
    sampling_method: Literal["spline", "contour_arc"],
    reporter: ImageProcessingIntermediateStepReporter,
) -> Path:
    with reporter.with_indent("tps_transform"):
        assert mask_image_path.exists()
        assert original_image_path.exists()
        original_image = cv2.imread(str(original_image_path))
        mask_binary = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        reporter.output("input_mask", mask_binary)
        reporter.output("input_image", original_image)

        src_points, dst_points = prepare_tps_points_from_mask(
            mask_binary,
            points_per_side=points_per_side,
            smoothing=smoothing,
            sampling_method=sampling_method,
            reporter=reporter,
        )

        plot_tps_input(original_image, src_points, dst_points, reporter=reporter)

        width = int(np.max(dst_points[:, 0]))
        height = int(np.max(dst_points[:, 1]))
        output_shape = (height, width)

        transformed_image = apply_tps_transform_skimage(
            original_image, src_points, dst_points, output_shape=output_shape
        )

        plot_original_transformed_image(original_image, transformed_image, reporter)
        return reporter.output("output_image", transformed_image, "output")


def plot_original_transformed_image(original_image: Img, transformed_image: Img, reporter: ImageProcessingIntermediateStepReporter):
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.title("Transformed Image")
    plt.axis("off")
    plt.tight_layout()
    reporter.output_mpl("original_transformed", plt.gcf())



def plot_tps_input(
    original_image: Img, src_points: Contour, dst_points: Contour, reporter: ImageProcessingIntermediateStepReporter
):
    # Now, `src_points` and `dst_points` are ready for TPS transformation
    print("Source Points:\n", len(src_points))
    print("Destination Points:\n", len(dst_points))
    # Plot the points for visualization
    plt.figure(figsize=(8, 8))
    # plt.imshow(mask_binary, cmap='gray', alpha=0.5, zorder=-1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), alpha=0.5, zorder=-1)
    plt.scatter(src_points[:, 0], src_points[:, 1], color="blue", label="Source Points", s=2)
    plt.scatter(dst_points[:, 0], dst_points[:, 1], color="red", label="Destination Points", s=2)
    # Plot a line from each source point to its corresponding destination point
    for src, dst in zip(src_points, dst_points):
        plt.plot([src[0], dst[0]], [src[1], dst[1]], color="green", linewidth=1)
    plt.legend()
    plt.title("Source and Destination Points for TPS")
    plt.axis("equal")
    plt.tight_layout()
    reporter.output_mpl("tps_input", plt.gcf())

