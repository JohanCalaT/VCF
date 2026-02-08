"""
Temporal filtering using lifting scheme with motion compensation.

This module implements temporal wavelet filtering for MCTF using
the lifting scheme with Predict and Update steps.

References:
    - Pesquet-Popescu, B., Bottreau, V. (2001). "Three-dimensional lifting 
      schemes for motion compensated video compression"
    - Secker, A., Taubman, D. (2003). "Lifting-based invertible motion 
      adaptive transform (LIMAT) framework"
    - Gonzalez-Ruiz, V. "MCTF"
      https://github.com/vicente-gonzalez-ruiz/motion_compensated_temporal_filtering
"""

import numpy as np
from typing import List, Tuple
import logging

from motion_compensation import motion_compensate


# Wavelet coefficients for lifting scheme
WAVELET_COEFFICIENTS = {
    'haar': {
        'predict': 1.0,
        'update': 0.5
    },
    '5/3': {
        'predict': 0.5,
        'update': 0.25
    },
    '9/7': {
        'predict': 1.586134342,
        'update': 0.052980118
    }
}


def get_wavelet_coefficients(wavelet_type: str) -> Tuple[float, float]:
    """
    Get predict and update coefficients for a wavelet.

    Args:
        wavelet_type: Wavelet type ('haar', '5/3', '9/7')

    Returns:
        (predict_coef, update_coef): Lifting scheme coefficients

    Raises:
        ValueError: If wavelet type is not supported
    """
    if wavelet_type not in WAVELET_COEFFICIENTS:
        raise ValueError(
            f"Wavelet type '{wavelet_type}' not supported. "
            f"Supported types: {list(WAVELET_COEFFICIENTS.keys())}"
        )

    coeffs = WAVELET_COEFFICIENTS[wavelet_type]
    return coeffs['predict'], coeffs['update']


def temporal_filter_lifting(
    frames: List[np.ndarray],
    motion_vectors_forward: List[np.ndarray],
    motion_vectors_backward: List[np.ndarray],
    wavelet_type: str = '5/3',
    block_size: int = 16,
    return_predictions: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply temporal filtering using lifting scheme with motion compensation.

    Implements IBB... scheme where even frames are low-pass (L)
    and odd frames generate high-pass (H) as prediction residuals.

    Args:
        frames: List of frames to filter
        motion_vectors_forward: List of forward MVs
        motion_vectors_backward: List of backward MVs
        wavelet_type: Wavelet type ('haar', '5/3', '9/7')
        block_size: Block size for MC
        return_predictions: If True, also return prediction images

    Returns:
        low_pass: Low temporal frequency frames (L)
        high_pass: High temporal frequency frames (H/residuals)
        predictions: Prediction images (only if return_predictions=True)
    """

    n_frames = len(frames)
    predict_coef, update_coef = get_wavelet_coefficients(wavelet_type)

    logging.info(f"Temporal filtering {n_frames} frames with {wavelet_type} wavelet")

    low_pass = []
    high_pass = []
    predictions = []

    # Process frame pairs (even, odd)
    for i in range(0, n_frames - 1, 2):
        frame_even = frames[i].astype(np.float32)      # Even frame (t=0,2,4...)
        frame_odd = frames[i + 1].astype(np.float32)   # Odd frame (t=1,3,5...)

        # === PREDICT STEP ===
        # Predict odd frame using MC from neighboring even frames

        # MC from previous frame (current even)
        mc_prev = motion_compensate(
            frame_even,
            motion_vectors_backward[i] if i < len(motion_vectors_backward) else np.zeros_like(motion_vectors_backward[0]),
            block_size
        )

        # MC from next frame (even i+2) if exists
        if i + 2 < n_frames:
            mc_next = motion_compensate(
                frames[i + 2].astype(np.float32),
                motion_vectors_forward[i + 1] if i + 1 < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
                block_size
            )
        else:
            mc_next = mc_prev

        # Bidirectional prediction
        prediction = (mc_prev + mc_next) * predict_coef / 2.0

        # Store prediction if requested
        if return_predictions:
            predictions.append(prediction)

        # High frequency residual (H)
        h_frame = frame_odd - prediction
        high_pass.append(h_frame)

        # === UPDATE STEP ===
        # Update even frame with residual information
        mc_residual = motion_compensate(
            h_frame,
            motion_vectors_forward[i] if i < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
            block_size
        )

        # Update (L)
        l_frame = frame_even + mc_residual * update_coef
        low_pass.append(l_frame)

    # If odd number of frames, last one passes as low-pass
    if n_frames % 2 != 0:
        low_pass.append(frames[-1].astype(np.float32))

    logging.info(f"Temporal filtering complete: {len(low_pass)} L frames, {len(high_pass)} H frames")

    if return_predictions:
        return low_pass, high_pass, predictions
    return low_pass, high_pass


def inverse_temporal_filter_lifting(
    low_pass: List[np.ndarray],
    high_pass: List[np.ndarray],
    motion_vectors_forward: List[np.ndarray],
    motion_vectors_backward: List[np.ndarray],
    wavelet_type: str = '5/3',
    block_size: int = 16
) -> List[np.ndarray]:
    """
    Reconstruct frames from temporal decomposition.

    Applies inverse lifting scheme to recover original frames
    from L (low-pass) and H (high-pass) components.

    Args:
        low_pass: L frames (low temporal frequency)
        high_pass: H frames (high temporal frequency)
        motion_vectors_forward: Forward MVs
        motion_vectors_backward: Backward MVs
        wavelet_type: Wavelet type used in encoding
        block_size: Block size for MC

    Returns:
        reconstructed_frames: List of reconstructed frames
    """

    predict_coef, update_coef = get_wavelet_coefficients(wavelet_type)

    n_low = len(low_pass)
    n_high = len(high_pass)

    logging.info(f"Inverse temporal filtering: {n_low} L frames, {n_high} H frames")

    reconstructed_frames = []

    for i in range(n_high):
        l_frame = low_pass[i].astype(np.float32)
        h_frame = high_pass[i].astype(np.float32)

        # === INVERSE UPDATE ===
        # Recover original even frame
        mc_residual = motion_compensate(
            h_frame,
            motion_vectors_forward[2 * i] if 2 * i < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
            block_size
        )

        frame_even = l_frame - mc_residual * update_coef
        reconstructed_frames.append(frame_even)

        # === INVERSE PREDICT ===
        # Recover original odd frame

        # MC for prediction
        mc_prev = motion_compensate(
            frame_even,
            motion_vectors_backward[2 * i] if 2 * i < len(motion_vectors_backward) else np.zeros_like(motion_vectors_backward[0]),
            block_size
        )

        if i + 1 < n_low:
            mc_next = motion_compensate(
                low_pass[i + 1].astype(np.float32),
                motion_vectors_forward[2 * i + 1] if 2 * i + 1 < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
                block_size
            )
        else:
            mc_next = mc_prev

        # Bidirectional prediction
        prediction = (mc_prev + mc_next) * predict_coef / 2.0

        # Inverse predict
        frame_odd = h_frame + prediction
        reconstructed_frames.append(frame_odd)

    # If there was an extra frame in low_pass (odd number of frames)
    if n_low > n_high:
        reconstructed_frames.append(low_pass[-1].astype(np.float32))

    logging.info(f"Inverse temporal filtering complete: {len(reconstructed_frames)} frames")

    return reconstructed_frames
