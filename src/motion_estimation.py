"""
Bidirectional motion estimation using block matching.

This module implements bidirectional motion estimation required
for MCTF (Motion Compensated Temporal Filtering).

Implements Diamond Search algorithm for fast motion estimation,
which is much faster than Full Search while maintaining good quality.

References:
    - Zhu, S., Ma, K.K. (2000). "A new diamond search algorithm for fast block-matching"
    - Ohm, J.R. (1994). "Three-dimensional subband coding with motion compensation"
    - Gonzalez-Ruiz, V. "MCTF" https://github.com/vicente-gonzalez-ruiz/motion_compensated_temporal_filtering
"""

import numpy as np
from typing import Tuple
import logging


# Diamond search patterns
# Large Diamond Search Pattern (LDSP) - 9 points
LDSP = np.array([
    [0, 0],    # center
    [0, -2],   # top
    [0, 2],    # bottom
    [-2, 0],   # left
    [2, 0],    # right
    [-1, -1],  # top-left
    [1, -1],   # top-right
    [-1, 1],   # bottom-left
    [1, 1],    # bottom-right
], dtype=np.int32)

# Small Diamond Search Pattern (SDSP) - 5 points
SDSP = np.array([
    [0, 0],    # center
    [0, -1],   # top
    [0, 1],    # bottom
    [-1, 0],   # left
    [1, 0],    # right
], dtype=np.int32)


def block_matching_bidirectional(
    frame_current: np.ndarray,
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    block_size: int = 16,
    search_range: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bidirectional motion estimation using Diamond Search block matching.

    Uses Diamond Search algorithm which is O(log n) instead of O(n²),
    providing 10-50x speedup over Full Search with similar quality.

    Args:
        frame_current: Current frame (the one being predicted)
        frame_prev: Previous frame (backward reference)
        frame_next: Next frame (forward reference)
        block_size: Block size NxN (default: 16)
        search_range: Search range ±pixels (default: 16)

    Returns:
        mv_forward: Forward motion vectors (current→next) [blocks_h, blocks_w, 2]
        mv_backward: Backward motion vectors (current→prev) [blocks_h, blocks_w, 2]
    """

    height, width = frame_current.shape[:2]

    # Number of blocks in each dimension
    blocks_h = height // block_size
    blocks_w = width // block_size

    # Initialize motion vector fields
    mv_forward = np.zeros((blocks_h, blocks_w, 2), dtype=np.float32)
    mv_backward = np.zeros((blocks_h, blocks_w, 2), dtype=np.float32)

    logging.debug(f"Diamond Search: {blocks_h}x{blocks_w} blocks, size={block_size}, range={search_range}")

    # Convert frames to float32 once for faster computation
    frame_current_f = frame_current.astype(np.float32)
    frame_prev_f = frame_prev.astype(np.float32)
    frame_next_f = frame_next.astype(np.float32)

    # Iterate over each block
    for by in range(blocks_h):
        for bx in range(blocks_w):
            # Current block coordinates
            y_start = by * block_size
            x_start = bx * block_size
            y_end = y_start + block_size
            x_end = x_start + block_size

            # Extract current block
            current_block = frame_current_f[y_start:y_end, x_start:x_end]

            # Forward search (current → next)
            mv_forward[by, bx] = _diamond_search(
                current_block, frame_next_f,
                y_start, x_start, block_size,
                height, width, search_range
            )

            # Backward search (current → prev)
            mv_backward[by, bx] = _diamond_search(
                current_block, frame_prev_f,
                y_start, x_start, block_size,
                height, width, search_range
            )

    return mv_forward, mv_backward


def _diamond_search(
    current_block: np.ndarray,
    reference_frame: np.ndarray,
    y_start: int,
    x_start: int,
    block_size: int,
    height: int,
    width: int,
    search_range: int
) -> Tuple[float, float]:
    """
    Diamond Search algorithm for fast block matching.

    Two-step process:
    1. Large Diamond Search Pattern (LDSP) - coarse search
    2. Small Diamond Search Pattern (SDSP) - fine refinement

    Args:
        current_block: Block to search for
        reference_frame: Frame to search in
        y_start, x_start: Block position in original frame
        block_size: Block size
        height, width: Frame dimensions
        search_range: Maximum search range

    Returns:
        (dx, dy): Motion vector that minimizes SAD
    """

    # Start at center (0, 0)
    center_x, center_y = 0, 0
    best_mv = (0, 0)
    min_sad = _compute_sad(current_block, reference_frame,
                           y_start, x_start, 0, 0,
                           block_size, height, width)

    # Phase 1: Large Diamond Search Pattern
    max_iterations = search_range  # Limit iterations

    for iteration in range(max_iterations):
        found_better = False

        for i in range(1, len(LDSP)):  # Skip center (already computed)
            dx = center_x + LDSP[i, 0]
            dy = center_y + LDSP[i, 1]

            # Check if within search range
            if abs(dx) > search_range or abs(dy) > search_range:
                continue

            sad = _compute_sad(current_block, reference_frame,
                              y_start, x_start, dx, dy,
                              block_size, height, width)

            if sad < min_sad:
                min_sad = sad
                best_mv = (dx, dy)
                found_better = True

        # If best point is center, switch to SDSP
        if not found_better:
            break

        # Move center to best point
        center_x, center_y = best_mv

    # Phase 2: Small Diamond Search Pattern (refinement)
    for _ in range(2):  # A few refinement iterations
        found_better = False

        for i in range(1, len(SDSP)):  # Skip center
            dx = center_x + SDSP[i, 0]
            dy = center_y + SDSP[i, 1]

            # Check if within search range
            if abs(dx) > search_range or abs(dy) > search_range:
                continue

            sad = _compute_sad(current_block, reference_frame,
                              y_start, x_start, dx, dy,
                              block_size, height, width)

            if sad < min_sad:
                min_sad = sad
                best_mv = (dx, dy)
                found_better = True

        if not found_better:
            break

        center_x, center_y = best_mv

    return best_mv


def _compute_sad(
    current_block: np.ndarray,
    reference_frame: np.ndarray,
    y_start: int,
    x_start: int,
    dx: int,
    dy: int,
    block_size: int,
    height: int,
    width: int
) -> float:
    """
    Compute Sum of Absolute Differences (SAD) for a motion vector.

    Returns infinity if the reference block is out of bounds.
    """
    ref_y = y_start + dy
    ref_x = x_start + dx

    # Check bounds
    if (ref_y < 0 or ref_y + block_size > height or
        ref_x < 0 or ref_x + block_size > width):
        return float('inf')

    # Extract reference block and compute SAD
    ref_block = reference_frame[ref_y:ref_y + block_size,
                                ref_x:ref_x + block_size]

    return np.sum(np.abs(current_block - ref_block))
