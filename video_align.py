"""
Align video timestamps with KQHiveMind game data by detecting QR codes
in post-game summary screens.
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import cv2
from PIL import Image
from pyzbar.pyzbar import decode
from tqdm import tqdm


# Hardcoded delay between game end (victory event) and QR code appearance on stats screen
QR_DELAY_SECONDS = 18.3

# Cache directory for QR detection results
QR_CACHE_DIR = Path(__file__).parent / "cache" / "qr_detections"


def _get_qr_cache_path(video_path: str) -> Path:
    """Get cache file path for a video's QR detection."""
    video_name = Path(video_path).name
    return QR_CACHE_DIR / f"{video_name}.json"


def _load_qr_from_cache(video_path: str, verbose: bool = True) -> 'QRDetection | None':
    """Load QR detection from cache if available."""
    cache_path = _get_qr_cache_path(video_path)
    if not cache_path.exists():
        return None

    if verbose:
        print(f"Loading QR detection from cache: {cache_path}")

    with open(cache_path, 'r') as f:
        data = json.load(f)

    return QRDetection(
        game_id=data['game_id'],
        first_frame=data['first_frame'],
        detection_frame=data['detection_frame'],
    )


def _save_qr_to_cache(video_path: str, detection: 'QRDetection', verbose: bool = True) -> None:
    """Save QR detection to cache."""
    QR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_qr_cache_path(video_path)

    data = {
        'game_id': detection.game_id,
        'first_frame': detection.first_frame,
        'detection_frame': detection.detection_frame,
    }

    with open(cache_path, 'w') as f:
        json.dump(data, f)

    if verbose:
        print(f"Saved QR detection to cache: {cache_path}")


@contextmanager
def suppress_stderr():
    """Suppress stderr to hide noisy zbar warnings."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)


@dataclass
class QRDetection:
    """Represents a detected QR code and when it first appeared."""
    game_id: int
    first_frame: int
    detection_frame: int  # Frame where we initially spotted it


@dataclass
class VideoOffset:
    """Represents the calculated offset between video time and HiveMind UTC time."""
    video_start_utc: datetime
    game_id: int
    qr_first_frame: int
    fps: float


def calculate_video_offset(
    video_path: str,
    hivemind_victory_utc: datetime,
    qr_detection: QRDetection | None = None,
    verbose: bool = True
) -> VideoOffset:
    """
    Calculate the time offset between video timestamps and HiveMind UTC.

    Uses a hardcoded delay (QR_DELAY_SECONDS) between the game end event
    and when the QR code appears on the stats screen.

    Args:
        video_path: Path to video file
        hivemind_victory_utc: UTC timestamp of victory event from HiveMind
        qr_detection: Pre-computed QR detection (will find if None)
        verbose: Print progress

    Returns:
        VideoOffset with timing information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Find QR code if not provided
    if qr_detection is None:
        qr_detection = find_first_qr(video_path, verbose=verbose)
        if qr_detection is None:
            raise ValueError("No QR code found in video")

    # QR appears QR_DELAY_SECONDS after the actual victory
    # qr_video_time = victory_video_time + QR_DELAY_SECONDS
    # victory_video_time = qr_video_time - QR_DELAY_SECONDS
    qr_video_time = qr_detection.first_frame / fps
    victory_video_time = qr_video_time - QR_DELAY_SECONDS

    # video_start_utc + victory_video_time = hivemind_victory_utc
    video_start_utc = hivemind_victory_utc - timedelta(seconds=victory_video_time)

    if verbose:
        print(f"\nOffset calculation for game {qr_detection.game_id}:")
        print(f"  QR first appears at frame {qr_detection.first_frame} ({qr_video_time:.2f}s)")
        print(f"  Using {QR_DELAY_SECONDS}s delay, victory at {victory_video_time:.2f}s in video")
        print(f"  HiveMind victory time: {hivemind_victory_utc}")
        print(f"  Video start time (UTC): {video_start_utc}")

    return VideoOffset(
        video_start_utc=video_start_utc,
        game_id=qr_detection.game_id,
        qr_first_frame=qr_detection.first_frame,
        fps=fps,
    )


def hivemind_to_video_frame(utc_time: datetime, offset: VideoOffset) -> int:
    """Convert a HiveMind UTC timestamp to a video frame number."""
    elapsed = (utc_time - offset.video_start_utc).total_seconds()
    return int(elapsed * offset.fps)


def video_frame_to_hivemind(frame: int, offset: VideoOffset) -> datetime:
    """Convert a video frame number to HiveMind UTC timestamp."""
    elapsed_seconds = frame / offset.fps
    return offset.video_start_utc + timedelta(seconds=elapsed_seconds)


def extract_game_id_from_url(url: str) -> int | None:
    """Extract game ID from a KQHiveMind URL."""
    match = re.search(r'kqhivemind\.com/game/(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def detect_hivemind_qr(frame) -> int | None:
    """
    Detect a KQHiveMind QR code in a frame.
    Returns the game ID if found, None otherwise.
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    with suppress_stderr():
        qr_codes = decode(pil_image)
    for qr in qr_codes:
        try:
            data = qr.data.decode('utf-8')
            game_id = extract_game_id_from_url(data)
            if game_id is not None:
                return game_id
        except (UnicodeDecodeError, AttributeError):
            continue
    return None


def read_frame(cap: cv2.VideoCapture, frame_idx: int):
    """Read a specific frame from the video."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def binary_search_qr_start(
    cap: cv2.VideoCapture,
    game_id: int,
    found_frame: int,
    search_back_frames: int = 600,  # 10 seconds at 60fps
    verbose: bool = True
) -> int:
    """
    Binary search backwards to find the first frame where the QR code appears.

    Args:
        cap: OpenCV video capture
        game_id: The game ID we're looking for
        found_frame: Frame where we initially detected the QR
        search_back_frames: How far back to search (default 10 seconds)
        verbose: Print progress

    Returns:
        The first frame index where the QR code appears
    """
    # First, find a frame where the QR doesn't exist (lower bound)
    low = max(0, found_frame - search_back_frames)
    high = found_frame

    # Verify low doesn't have the QR
    frame = read_frame(cap, low)
    if frame is not None and detect_hivemind_qr(frame) == game_id:
        # QR exists even at low bound, search further back
        while low > 0:
            low = max(0, low - search_back_frames)
            frame = read_frame(cap, low)
            if frame is None or detect_hivemind_qr(frame) != game_id:
                break

    if verbose:
        print(f"  Binary searching between frames {low} and {high}")

    # Binary search for the first frame with the QR
    while low < high:
        mid = (low + high) // 2
        frame = read_frame(cap, mid)

        if frame is not None and detect_hivemind_qr(frame) == game_id:
            high = mid  # QR exists, search earlier
        else:
            low = mid + 1  # QR doesn't exist, search later

    return low


def find_first_qr(
    video_path: str,
    coarse_interval_sec: float = 5.0,
    use_cache: bool = True,
    verbose: bool = True
) -> QRDetection | None:
    """
    Find the first QR code in a video and determine exactly when it appeared.

    Strategy:
    1. Check cache first
    2. Coarse scan: Check every 5 seconds until we find a QR code
    3. Fine scan: Binary search backwards to find the exact first frame

    Args:
        video_path: Path to the video file
        coarse_interval_sec: Seconds between coarse scan samples (default 5s)
        use_cache: Whether to use cached results if available
        verbose: Print progress information

    Returns:
        QRDetection with game_id and first_frame, or None if no QR found
    """
    # Check cache first
    if use_cache:
        cached = _load_qr_from_cache(video_path, verbose=verbose)
        if cached is not None:
            return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    coarse_interval = int(coarse_interval_sec * fps)

    if verbose:
        duration_mins = total_frames / fps / 60
        print(f"Video: {total_frames} frames, {fps:.1f} fps, {duration_mins:.1f} minutes")
        print(f"Coarse scan every {coarse_interval_sec}s ({coarse_interval} frames)")

    # Phase 1: Coarse scan to find any QR code
    frame_indices = range(0, total_frames, coarse_interval)
    pbar = tqdm(frame_indices, desc="Coarse scan", disable=not verbose, unit="samples")

    found_frame = None
    game_id = None

    for frame_idx in pbar:
        frame = read_frame(cap, frame_idx)
        if frame is None:
            break

        game_id = detect_hivemind_qr(frame)
        if game_id is not None:
            found_frame = frame_idx
            pbar.set_postfix({"found": game_id})
            break

    pbar.close()

    if found_frame is None:
        if verbose:
            print("No QR codes found")
        cap.release()
        return None

    if verbose:
        time_sec = found_frame / fps
        print(f"Found game {game_id} at frame {found_frame} ({time_sec/60:.0f}:{time_sec%60:05.2f})")
        print("Binary searching for exact start frame...")

    # Phase 2: Binary search to find exact first frame
    first_frame = binary_search_qr_start(cap, game_id, found_frame, verbose=verbose)

    cap.release()

    if verbose:
        time_sec = first_frame / fps
        print(f"QR code first appears at frame {first_frame} ({time_sec/60:.0f}:{time_sec%60:05.2f})")

    result = QRDetection(
        game_id=game_id,
        first_frame=first_frame,
        detection_frame=found_frame
    )

    # Save to cache for future runs
    if use_cache:
        _save_qr_to_cache(video_path, result, verbose=verbose)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_align.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    result = find_first_qr(video_path)

    if result:
        print(f"\nResult: game_id={result.game_id}, first_frame={result.first_frame}")
