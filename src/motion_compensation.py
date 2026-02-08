"""
Motion compensation for MCTF.

This module implements motion compensation that applies motion vectors
to reference frames to generate predictions.

References:
    - Gonzalez-Ruiz, V. "Motion Compensation"
      https://github.com/vicente-gonzalez-ruiz/motion_compensation
    - Pesquet-Popescu, B., Bottreau, V. (2001). "Three-dimensional lifting 
      schemes for motion compensated video compression"
"""

import numpy as np
import logging


def motion_compensate(
    frame: np.ndarray,
    motion_vectors: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """
    Apply motion compensation to a frame.

    Generates a compensated frame by shifting blocks according to
    the provided motion vectors.

    Args:
        frame: Reference frame
        motion_vectors: Motion vector field [blocks_h, blocks_w, 2]
        block_size: Block size (default: 16)

    Returns:
        compensated_frame: Motion compensated frame
    """

    height, width = frame.shape[:2]
    compensated_frame = np.zeros_like(frame, dtype=np.float32)

    blocks_h, blocks_w = motion_vectors.shape[:2]

    logging.debug(f"Motion compensate: {blocks_h}x{blocks_w} blocks, size={block_size}")

    for by in range(blocks_h):
        for bx in range(blocks_w):
            # Destination block coordinates
            y_start = by * block_size
            x_start = bx * block_size
            y_end = min(y_start + block_size, height)
            x_end = min(x_start + block_size, width)

            # Motion vector
            dx, dy = motion_vectors[by, bx]

            # Coordinates in reference frame
            ref_y = int(y_start + dy)
            ref_x = int(x_start + dx)

            # Check boundaries
            if (ref_y >= 0 and ref_y + block_size <= height and
                ref_x >= 0 and ref_x + block_size <= width):

                # Copy compensated block
                compensated_frame[y_start:y_end, x_start:x_end] = \
                    frame[ref_y:ref_y + (y_end - y_start), 
                          ref_x:ref_x + (x_end - x_start)]
            else:
                # If out of bounds, copy original block
                compensated_frame[y_start:y_end, x_start:x_end] = \
                    frame[y_start:y_end, x_start:x_end]

    return compensated_frame


def motion_compensate_bidirectional(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    mv_backward: np.ndarray,
    mv_forward: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """
    Bidirectional motion compensation for B-frames.

    Generates a bidirectional prediction by averaging compensations
    from previous and next frames.

    Args:
        frame_prev: Previous frame (backward reference)
        frame_next: Next frame (forward reference)
        mv_backward: Backward motion vectors
        mv_forward: Forward motion vectors
        block_size: Block size (default: 16)

    Returns:
        prediction: Bidirectional prediction frame
    """

    # Compensate from previous frame
    mc_prev = motion_compensate(frame_prev, mv_backward, block_size)

    # Compensate from next frame
    mc_next = motion_compensate(frame_next, mv_forward, block_size)

    # Bidirectional prediction (average)
    prediction = (mc_prev + mc_next) / 2.0

    return prediction
