"""
MCTF: Motion Compensated Temporal Filtering codec.

This module implements a video codec based on temporal filtering
with motion compensation using lifting scheme.

Key features:
- Bidirectional motion estimation (Diamond Search algorithm)
- Temporal filtering with Haar, 5/3, or 9/7 wavelets
- Block DCT with zigzag scan for spatial compression
- GOP-based processing
"""

import logging
import os
import numpy as np
import cv2
from PIL import Image
import pickle
import zlib

# Import VCF framework modules
import entropy_video_coding as EVC
import parser
import main
import platform_utils as pu

# Import MCTF components
from motion_estimation import block_matching_bidirectional
from motion_compensation import motion_compensate
from temporal_filtering import temporal_filter_lifting, inverse_temporal_filter_lifting

# Get temporary directory
TMP_DIR = pu.get_vcf_temp_dir()

# Default parameters
DEFAULT_GOP_SIZE = 16
DEFAULT_TEMPORAL_LEVELS = 4
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
DEFAULT_WAVELET_TYPE = '5/3'
DEFAULT_QSS = 128  # Quantization step size (higher than 2D-DCT due to simpler entropy coding)

# Encoder parser
parser.parser_encode.add_argument("-V", "--video_input", type=parser.int_or_str,
    help=f"Input video (default: {EVC.ENCODE_INPUT})",
    default=EVC.ENCODE_INPUT)
parser.parser_encode.add_argument("-O", "--video_output", type=parser.int_or_str,
    help=f"Output prefix (default: {EVC.ENCODE_OUTPUT_PREFIX})",
    default=EVC.ENCODE_OUTPUT_PREFIX)
parser.parser_encode.add_argument("-T", "--transform", type=str,
    help=f"2D-transform (default: {EVC.DEFAULT_TRANSFORM})",
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_encode.add_argument("-N", "--number_of_frames", type=parser.int_or_str,
    help=f"Number of frames to encode (default: {EVC.N_FRAMES})",
    default=f"{EVC.N_FRAMES}")
# Note: -q/--QSS is already defined in deadzone.py (imported via 2D-DCT chain)
parser.parser_encode.add_argument("--gop_size", type=int,
    help=f"GOP size (default: {DEFAULT_GOP_SIZE})",
    default=DEFAULT_GOP_SIZE)
parser.parser_encode.add_argument("--temporal_levels", type=int,
    help=f"Temporal decomposition levels (default: {DEFAULT_TEMPORAL_LEVELS})",
    default=DEFAULT_TEMPORAL_LEVELS)
parser.parser_encode.add_argument("--block_size", type=int,
    help=f"Block size for motion estimation (default: {DEFAULT_BLOCK_SIZE})",
    default=DEFAULT_BLOCK_SIZE)
parser.parser_encode.add_argument("--search_range", type=int,
    help=f"Search range for motion estimation (default: {DEFAULT_SEARCH_RANGE})",
    default=DEFAULT_SEARCH_RANGE)
parser.parser_encode.add_argument("--wavelet_type", type=str,
    help=f"Temporal wavelet type: haar, 5/3, 9/7 (default: {DEFAULT_WAVELET_TYPE})",
    default=DEFAULT_WAVELET_TYPE)

# Decoder parser
parser.parser_decode.add_argument("-V", "--video_input", type=parser.int_or_str,
    help=f"Input MCTF stream prefix (default: {EVC.ENCODE_OUTPUT_PREFIX})",
    default=EVC.ENCODE_OUTPUT_PREFIX)
parser.parser_decode.add_argument("-O", "--video_output", type=parser.int_or_str,
    help=f"Output prefix (default: {EVC.DECODE_OUTPUT_PREFIX})",
    default=EVC.DECODE_OUTPUT_PREFIX)
parser.parser_decode.add_argument("-T", "--transform", type=str,
    help=f"2D-transform (default: {EVC.DEFAULT_TRANSFORM})",
    default=EVC.DEFAULT_TRANSFORM)
parser.parser_decode.add_argument("-N", "--number_of_frames", type=parser.int_or_str,
    help=f"Number of frames to decode (default: {EVC.N_FRAMES})",
    default=f"{EVC.N_FRAMES}")
# Note: -q/--QSS is already defined in deadzone.py (imported via 2D-DCT chain)
parser.parser_decode.add_argument("--gop_size", type=int,
    help=f"GOP size (default: {DEFAULT_GOP_SIZE})",
    default=DEFAULT_GOP_SIZE)
parser.parser_decode.add_argument("--temporal_levels", type=int,
    help=f"Temporal decomposition levels (default: {DEFAULT_TEMPORAL_LEVELS})",
    default=DEFAULT_TEMPORAL_LEVELS)
parser.parser_decode.add_argument("--block_size", type=int,
    help=f"Block size for motion estimation (default: {DEFAULT_BLOCK_SIZE})",
    default=DEFAULT_BLOCK_SIZE)
parser.parser_decode.add_argument("--search_range", type=int,
    help=f"Search range for motion estimation (default: {DEFAULT_SEARCH_RANGE})",
    default=DEFAULT_SEARCH_RANGE)
parser.parser_decode.add_argument("--wavelet_type", type=str,
    help=f"Temporal wavelet type: haar, 5/3, 9/7 (default: {DEFAULT_WAVELET_TYPE})",
    default=DEFAULT_WAVELET_TYPE)

# Import transform module
import importlib
args = parser.parser.parse_known_args()[0]
transform = importlib.import_module(args.transform)


class CoDec(transform.CoDec):
    """
    MCTF (Motion Compensated Temporal Filtering) Codec.

    Implements temporal filtering with motion compensation using
    the lifting scheme for video compression.

    Pipeline:
    1. Bidirectional motion estimation
    2. Temporal filtering with lifting scheme (Predict + Update)
    3. Quantization
    4. Entropy coding (zlib compression)
    """

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)

        # MCTF configuration
        self.gop_size = args.gop_size
        self.temporal_levels = args.temporal_levels
        self.block_size = args.block_size
        self.search_range = args.search_range
        self.wavelet_type = args.wavelet_type
        self.QSS = args.QSS

        logging.info(f"MCTF Codec initialized:")
        logging.info(f"  GOP size: {self.gop_size}")
        logging.info(f"  Temporal levels: {self.temporal_levels}")
        logging.info(f"  Block size: {self.block_size}")
        logging.info(f"  Search range: {self.search_range}")
        logging.info(f"  Wavelet type: {self.wavelet_type}")
        logging.info(f"  QSS: {self.QSS}")

    def bye(self):
        """Override bye() to use video_input/video_output."""
        logging.debug("trace")
        if __debug__:
            if self.encoding:
                BPP = (self.total_output_size * 8) / (self.N_frames * self.width * self.height)
                logging.info(f"Output bit-rate = {BPP} bits/pixel")
                with open(f"{self.args.video_output}.txt", 'w') as f:
                    f.write(f"{self.args.video_input}\n")
                    f.write(f"{self.N_frames}\n")
                    f.write(f"{self.height}\n")
                    f.write(f"{self.width}\n")
                    f.write(f"{BPP}\n")
            else:
                with open(f"{self.args.video_input}.txt", 'r') as f:
                    original_file = f.readline().strip()
                    logging.info(f"original_file = {original_file}")
                    N_frames = int(f.readline().strip())
                    logging.info(f"N_frames = {N_frames}")
                    height = f.readline().strip()
                    logging.info(f"video height = {height} pixels")
                    width = f.readline().strip()
                    logging.info(f"video width = {width} pixels")
                    BPP = float(f.readline().strip())
                    logging.info(f"BPP = {BPP}")

    def quantize(self, data):
        """Quantize data using QSS."""
        return np.round(data / self.QSS).astype(np.int16)

    def dequantize(self, data):
        """Dequantize data using QSS."""
        return (data * self.QSS).astype(np.float32)

    def compress(self, data):
        """Compress data using zlib."""
        return zlib.compress(data.tobytes(), level=9)

    def decompress(self, data, shape, dtype):
        """Decompress data using zlib."""
        decompressed = zlib.decompress(data)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)

    def zigzag_scan_block(self, block):
        """Apply zigzag scan to an 8x8 block (like JPEG)."""
        zigzag_order = [
            0,  1,  8, 16,  9,  2,  3, 10,
           17, 24, 32, 25, 18, 11,  4,  5,
           12, 19, 26, 33, 40, 48, 41, 34,
           27, 20, 13,  6,  7, 14, 21, 28,
           35, 42, 49, 56, 57, 50, 43, 36,
           29, 22, 15, 23, 30, 37, 44, 51,
           58, 59, 52, 45, 38, 31, 39, 46,
           53, 60, 61, 54, 47, 55, 62, 63
        ]
        flat = block.flatten()
        return np.array([flat[i] for i in zigzag_order])

    def inverse_zigzag_block(self, zigzag_data):
        """Inverse zigzag scan for an 8x8 block."""
        zigzag_order = [
            0,  1,  8, 16,  9,  2,  3, 10,
           17, 24, 32, 25, 18, 11,  4,  5,
           12, 19, 26, 33, 40, 48, 41, 34,
           27, 20, 13,  6,  7, 14, 21, 28,
           35, 42, 49, 56, 57, 50, 43, 36,
           29, 22, 15, 23, 30, 37, 44, 51,
           58, 59, 52, 45, 38, 31, 39, 46,
           53, 60, 61, 54, 47, 55, 62, 63
        ]
        block = np.zeros(64, dtype=zigzag_data.dtype)
        for i, idx in enumerate(zigzag_order):
            block[idx] = zigzag_data[i]
        return block.reshape(8, 8)

    def zigzag_frame(self, frame, block_size=8):
        """Apply zigzag scan to all blocks in a frame."""
        h, w = frame.shape
        result = []
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = frame[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    result.append(self.zigzag_scan_block(block))
        return np.concatenate(result)

    def inverse_zigzag_frame(self, zigzag_data, shape, block_size=8):
        """Inverse zigzag scan for a frame."""
        h, w = shape
        frame = np.zeros((h, w), dtype=zigzag_data.dtype)
        block_idx = 0
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if i + block_size <= h and j + block_size <= w:
                    block_data = zigzag_data[block_idx*64:(block_idx+1)*64]
                    frame[i:i+block_size, j:j+block_size] = self.inverse_zigzag_block(block_data)
                    block_idx += 1
        return frame

    def block_dct(self, frame, block_size=8):
        """Apply DCT to frame in blocks (like JPEG)."""
        h, w = frame.shape
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w)), mode='edge')

        result = np.zeros_like(frame, dtype=np.float32)
        for i in range(0, frame.shape[0], block_size):
            for j in range(0, frame.shape[1], block_size):
                block = frame[i:i+block_size, j:j+block_size].astype(np.float32)
                result[i:i+block_size, j:j+block_size] = cv2.dct(block)

        return result[:h, :w] if pad_h > 0 or pad_w > 0 else result

    def block_idct(self, dct_frame, block_size=8):
        """Apply inverse DCT to frame in blocks."""
        h, w = dct_frame.shape
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            dct_frame = np.pad(dct_frame, ((0, pad_h), (0, pad_w)), mode='constant')

        result = np.zeros_like(dct_frame, dtype=np.float32)
        for i in range(0, dct_frame.shape[0], block_size):
            for j in range(0, dct_frame.shape[1], block_size):
                block = dct_frame[i:i+block_size, j:j+block_size].astype(np.float32)
                result[i:i+block_size, j:j+block_size] = cv2.idct(block)

        return result[:h, :w] if pad_h > 0 or pad_w > 0 else result

    def encode(self):
        """
        Encode a video using MCTF.

        Process:
        1. Read frames from input video
        2. Group frames into GOPs
        3. For each GOP:
           a. Estimate bidirectional motion
           b. Apply temporal filtering (lifting)
           c. Quantize and compress L and H frames
        4. Write encoded stream
        """
        logging.debug("trace")
        self.encoding = True

        # Read input video
        input_path = self.args.video_input
        logging.info(f"MCTF Encoding {input_path}")

        # Read frames in color (RGB)
        import av
        container = av.open(input_path)
        frames_rgb = []
        N = int(self.args.number_of_frames)

        for frame in container.decode(video=0):
            if len(frames_rgb) >= N:
                break
            img = frame.to_ndarray(format='rgb24')
            frames_rgb.append(img)

            # Save original frame for comparison
            orig_fn = os.path.join(TMP_DIR, f"original_{len(frames_rgb)-1:04d}.png")
            Image.fromarray(img).save(orig_fn)

        container.close()

        self.N_frames = len(frames_rgb)
        self.height, self.width = frames_rgb[0].shape[:2]
        self.is_color = len(frames_rgb[0].shape) == 3
        logging.info(f"Read {self.N_frames} frames of size {self.width}x{self.height} (color={self.is_color})")

        # Convert RGB to YCbCr for processing
        # Y channel gets full MCTF, Cb/Cr get simpler processing
        frames_y = []
        frames_cb = []
        frames_cr = []

        for img in frames_rgb:
            # Convert RGB to YCbCr
            img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            frames_y.append(img_ycbcr[:, :, 0].astype(np.float32))   # Y
            frames_cb.append(img_ycbcr[:, :, 2].astype(np.float32))  # Cb (note: OpenCV uses YCrCb order)
            frames_cr.append(img_ycbcr[:, :, 1].astype(np.float32))  # Cr

        # Store chrominance for later
        self.frames_cb = frames_cb
        self.frames_cr = frames_cr

        # Use Y channel for MCTF processing
        frames = frames_y

        # Process GOPs
        gop_data_list = []
        frame_idx = 0
        gop_counter = 0

        while frame_idx < self.N_frames:
            gop_end = min(frame_idx + self.gop_size, self.N_frames)
            gop_frames = frames[frame_idx:gop_end]
            gop_cb = self.frames_cb[frame_idx:gop_end]
            gop_cr = self.frames_cr[frame_idx:gop_end]

            logging.info(f"Processing GOP {gop_counter}: frames {frame_idx}-{gop_end-1}")

            # Encode GOP (Y channel with full MCTF)
            gop_data = self._encode_gop(gop_frames, gop_counter)

            # Encode chrominance channels (simpler: just quantize and compress)
            compressed_cb = []
            compressed_cr = []
            for cb, cr in zip(gop_cb, gop_cr):
                # Cb channel
                cb_quantized = self.quantize(cb)
                cb_compressed = self.compress(cb_quantized)
                compressed_cb.append({'data': cb_compressed, 'shape': cb.shape})
                # Cr channel
                cr_quantized = self.quantize(cr)
                cr_compressed = self.compress(cr_quantized)
                compressed_cr.append({'data': cr_compressed, 'shape': cr.shape})

            gop_data['compressed_cb'] = compressed_cb
            gop_data['compressed_cr'] = compressed_cr
            gop_data_list.append(gop_data)

            frame_idx = gop_end
            gop_counter += 1

        # Write encoded stream (this is the ONLY output we count)
        self._write_mctf_stream(gop_data_list)

        logging.info(f"MCTF encoding complete: {gop_counter} GOPs")

    def _encode_gop(self, gop_frames, gop_idx):
        """Encode a GOP using MCTF."""
        n_frames = len(gop_frames)

        if n_frames < 2:
            return self._encode_intra_only(gop_frames, gop_idx)

        # === Step 1: Motion estimation ===
        logging.info(f"  Motion estimation for {n_frames} frames")
        mv_forward_list = []
        mv_backward_list = []

        for i in range(n_frames):
            frame_current = gop_frames[i]
            frame_prev = gop_frames[max(0, i - 1)]
            frame_next = gop_frames[min(n_frames - 1, i + 1)]

            mv_fwd, mv_bwd = block_matching_bidirectional(
                frame_current, frame_prev, frame_next,
                self.block_size, self.search_range
            )
            mv_forward_list.append(mv_fwd)
            mv_backward_list.append(mv_bwd)

        # === Step 2: Temporal filtering ===
        logging.info(f"  Temporal filtering with {self.wavelet_type} wavelet")
        low_pass, high_pass, predictions = temporal_filter_lifting(
            gop_frames,
            mv_forward_list,
            mv_backward_list,
            self.wavelet_type,
            self.block_size,
            return_predictions=True
        )

        # Save prediction images to disk
        for i, pred in enumerate(predictions):
            pred_uint8 = np.clip(pred, 0, 255).astype(np.uint8)
            pred_fn = os.path.join(TMP_DIR, f"prediction_{gop_idx:02d}_{i:04d}.png")
            Image.fromarray(pred_uint8).save(pred_fn)
        logging.info(f"  Saved {len(predictions)} prediction images")

        # === Step 3: Apply spatial transform, quantize and compress L and H frames ===
        logging.info(f"  Quantizing and compressing {len(low_pass)} L frames and {len(high_pass)} H frames")

        compressed_low = []
        for i, l_frame in enumerate(low_pass):
            # Apply block DCT (8x8 blocks like JPEG)
            l_dct = self.block_dct(l_frame)
            # Quantize DCT coefficients
            l_quantized = self.quantize(l_dct)
            # Apply zigzag scan (groups zeros together for better compression)
            l_zigzag = self.zigzag_frame(l_quantized)
            # Compress
            l_compressed = self.compress(l_zigzag)
            compressed_low.append({
                'data': l_compressed,
                'shape': l_frame.shape
            })

        compressed_high = []
        for i, h_frame in enumerate(high_pass):
            # Apply block DCT (8x8 blocks like JPEG)
            h_dct = self.block_dct(h_frame)
            # Quantize DCT coefficients
            h_quantized = self.quantize(h_dct)
            # Apply zigzag scan (groups zeros together for better compression)
            h_zigzag = self.zigzag_frame(h_quantized)
            # Compress
            h_compressed = self.compress(h_zigzag)
            compressed_high.append({
                'data': h_compressed,
                'shape': h_frame.shape
            })

        # Compress motion vectors
        mv_fwd_compressed = self.compress(np.array(mv_forward_list))
        mv_bwd_compressed = self.compress(np.array(mv_backward_list))

        return {
            'n_frames': n_frames,
            'mv_forward': mv_fwd_compressed,
            'mv_forward_shape': np.array(mv_forward_list).shape,
            'mv_backward': mv_bwd_compressed,
            'mv_backward_shape': np.array(mv_backward_list).shape,
            'compressed_low': compressed_low,
            'compressed_high': compressed_high,
            'wavelet_type': self.wavelet_type
        }

    def _encode_intra_only(self, frames, gop_idx):
        """Encode frames as intra only."""
        compressed_frames = []
        for frame in frames:
            f_quantized = self.quantize(frame.astype(np.float32))
            f_compressed = self.compress(f_quantized)
            compressed_frames.append({
                'data': f_compressed,
                'shape': frame.shape
            })

        return {
            'n_frames': len(frames),
            'intra_only': True,
            'compressed_frames': compressed_frames
        }

    def _write_mctf_stream(self, gop_data_list):
        """Write the encoded MCTF stream."""
        stream_fn = os.path.join(TMP_DIR, "encoded.mctf")

        header = {
            'n_frames': self.N_frames,
            'height': self.height,
            'width': self.width,
            'gop_size': self.gop_size,
            'temporal_levels': self.temporal_levels,
            'block_size': self.block_size,
            'wavelet_type': self.wavelet_type,
            'QSS': self.QSS,
            'num_gops': len(gop_data_list)
        }

        with open(stream_fn, 'wb') as f:
            pickle.dump(header, f)
            for gop_data in gop_data_list:
                pickle.dump(gop_data, f)

        stream_size = os.path.getsize(stream_fn)
        self.total_output_size = stream_size
        logging.info(f"Written MCTF stream: {stream_fn} ({stream_size} bytes)")

    def decode(self):
        """
        Decode a video using MCTF.

        Process:
        1. Read encoded stream
        2. For each GOP:
           a. Decompress and dequantize L and H frames
           b. Apply inverse temporal filtering
           c. Reconstruct frames
        3. Write decoded frames
        """
        logging.debug("trace")
        self.encoding = False

        stream_fn = os.path.join(TMP_DIR, "encoded.mctf")
        logging.info(f"MCTF Decoding {self.args.video_input}")

        # Read encoded stream
        header, gop_data_list = self._read_mctf_stream(stream_fn)

        self.N_frames = header['n_frames']
        self.height = header['height']
        self.width = header['width']
        self.QSS = header.get('QSS', DEFAULT_QSS)

        logging.info(f"Decoding {self.N_frames} frames of size {self.width}x{self.height}")
        logging.info(f"  {header['num_gops']} GOPs, QSS={self.QSS}")

        # Decode each GOP
        all_frames_y = []
        all_frames_cb = []
        all_frames_cr = []

        for gop_idx, gop_data in enumerate(gop_data_list):
            logging.info(f"Decoding GOP {gop_idx}")

            if gop_data.get('intra_only', False):
                gop_frames_y = self._decode_intra_only(gop_data)
            else:
                gop_frames_y = self._decode_gop(gop_data)

            all_frames_y.extend(gop_frames_y)

            # Decode chrominance channels
            if 'compressed_cb' in gop_data:
                for cb_item, cr_item in zip(gop_data['compressed_cb'], gop_data['compressed_cr']):
                    cb_quantized = self.decompress(cb_item['data'], cb_item['shape'], np.int16)
                    cb_frame = self.dequantize(cb_quantized)
                    all_frames_cb.append(cb_frame)

                    cr_quantized = self.decompress(cr_item['data'], cr_item['shape'], np.int16)
                    cr_frame = self.dequantize(cr_quantized)
                    all_frames_cr.append(cr_frame)

        # Write decoded frames (in color if chrominance available)
        is_color = len(all_frames_cb) > 0

        for i, frame_y in enumerate(all_frames_y):
            if is_color and i < len(all_frames_cb):
                # Reconstruct color frame
                y = np.clip(frame_y, 0, 255).astype(np.uint8)
                cb = np.clip(all_frames_cb[i], 0, 255).astype(np.uint8)
                cr = np.clip(all_frames_cr[i], 0, 255).astype(np.uint8)

                # Combine YCrCb (OpenCV order)
                ycrcb = np.stack([y, cr, cb], axis=-1)
                rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

                out_fn = os.path.join(TMP_DIR, f"decoded_{i:04d}.png")
                Image.fromarray(rgb).save(out_fn)
            else:
                # Grayscale
                frame_uint8 = np.clip(frame_y, 0, 255).astype(np.uint8)
                out_fn = os.path.join(TMP_DIR, f"decoded_{i:04d}.png")
                Image.fromarray(frame_uint8).save(out_fn)

            logging.info(f"Decoded frame {i} to {out_fn}")

        logging.info(f"MCTF decoding complete: {len(all_frames_y)} frames (color={is_color})")

    def _read_mctf_stream(self, stream_fn):
        """Read the encoded MCTF stream."""
        with open(stream_fn, 'rb') as f:
            header = pickle.load(f)
            gop_data_list = []
            for _ in range(header['num_gops']):
                gop_data = pickle.load(f)
                gop_data_list.append(gop_data)

        return header, gop_data_list

    def _decode_gop(self, gop_data):
        """Decode a GOP using inverse MCTF."""
        n_frames = gop_data['n_frames']
        wavelet_type = gop_data['wavelet_type']

        # === Step 1: Decompress, dequantize and apply inverse DCT to L and H frames ===
        logging.info(f"  Decompressing L and H frames")

        low_pass = []
        for item in gop_data['compressed_low']:
            shape = item['shape']
            n_blocks = (shape[0] // 8) * (shape[1] // 8)
            zigzag_shape = (n_blocks * 64,)
            l_zigzag = self.decompress(item['data'], zigzag_shape, np.int16)
            l_quantized = self.inverse_zigzag_frame(l_zigzag, shape)
            l_dct = self.dequantize(l_quantized)
            l_frame = self.block_idct(l_dct)
            low_pass.append(l_frame)

        high_pass = []
        for item in gop_data['compressed_high']:
            shape = item['shape']
            n_blocks = (shape[0] // 8) * (shape[1] // 8)
            zigzag_shape = (n_blocks * 64,)
            h_zigzag = self.decompress(item['data'], zigzag_shape, np.int16)
            h_quantized = self.inverse_zigzag_frame(h_zigzag, shape)
            h_dct = self.dequantize(h_quantized)
            h_frame = self.block_idct(h_dct)
            high_pass.append(h_frame)

        # Decompress motion vectors
        mv_forward = self.decompress(
            gop_data['mv_forward'],
            gop_data['mv_forward_shape'],
            np.float32
        )
        mv_backward = self.decompress(
            gop_data['mv_backward'],
            gop_data['mv_backward_shape'],
            np.float32
        )

        # === Step 2: Inverse temporal filtering ===
        logging.info(f"  Inverse temporal filtering with {wavelet_type} wavelet")
        reconstructed = inverse_temporal_filter_lifting(
            low_pass,
            high_pass,
            mv_forward,
            mv_backward,
            wavelet_type,
            self.block_size
        )

        return reconstructed

    def _decode_intra_only(self, gop_data):
        """Decode intra-only frames."""
        frames = []
        for item in gop_data['compressed_frames']:
            f_quantized = self.decompress(item['data'], item['shape'], np.int16)
            frame = self.dequantize(f_quantized)
            frames.append(frame)
        return frames


if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
