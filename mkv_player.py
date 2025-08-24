#!/usr/bin/env python3
"""
Simple MKV video player with subtitle overlay and subtitle‑free screen capture.

This script uses the OpenCV library to open and play back a video file (such as
an MKV).  It optionally overlays soft subtitles from an external `.srt` file
onto each frame for display.  When the user presses the **s** key during
playback, the program saves a screenshot of the current frame *without*
subtitles to disk.  The screenshot filename includes a timestamp so that
multiple captures do not overwrite each other.  The viewer also supports
pausing and resuming playback with the space bar and exiting with **q** or
Escape.

Dependencies
------------
This program depends only on packages that are typically available in a basic
Python distribution:

* OpenCV (`cv2`) for video decoding and display.
* NumPy for array manipulations (comes bundled with OpenCV on many platforms).

Neither `python‑vlc` nor a GUI toolkit such as Tkinter are required.  Audio
playback is not implemented because it would add additional dependencies.  The
focus here is on video frames and subtitle overlay.

Usage
-----
Run the program from the command line and provide the path to the video file
with the `--video` option.  You can optionally specify an external subtitle
file with `--subtitle`.  If no subtitle file is given, the program will look
for a `.srt` file with the same base name as the video in the same folder.

Example:

    python mkv_player.py --video mymovie.mkv --subtitle mymovie.srt

Key bindings while the player window is open:

* **Space** – pause/resume playback.
* **s** – save a screenshot of the current frame without subtitles.
* **q** or **Esc** – quit the player.

Implementation Notes
--------------------
OpenCV’s `VideoCapture` class opens a video file and provides sequential
access to its frames.  The first argument to `VideoCapture()` is the
file path, and OpenCV’s documentation notes that passing a string creates a
capture object that reads the video file frame by frame【5417716936850†L145-L166】.  Once the
capture is opened, calling `read()` retrieves each frame for display【5417716936850†L184-L188】.
After extracting a frame, it can be shown with `imshow()` and `waitKey()` can
pause execution to handle user input; when the window is closed
`destroyAllWindows()` and `cap.release()` free resources【5417716936850†L270-L274】.  When
the user requests a screenshot we bypass the subtitle overlay and write the
raw frame to disk using `cv2.imwrite`.  If a user needs to capture frames
without subtitles using the MPV media player, the MPV manual explains that
pressing **S** takes a screenshot without subtitles whereas **s** takes a
regular screenshot【47430801850246†L185-L188】.  This script achieves a
similar “no‑subtitle” capture by saving the raw frame before drawing any
overlay text.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np


def parse_timecode(time_str: str) -> float:
    """Convert an SRT timecode (HH:MM:SS,mmm) into seconds as a float."""
    try:
        hms, ms = time_str.split(",")
        hours, minutes, seconds = map(int, hms.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds + int(ms) / 1000.0
        return total_seconds
    except Exception as exc:
        raise ValueError(f"Invalid timecode format: {time_str}") from exc


def parse_srt(path: str) -> List[Tuple[float, float, List[str]]]:
    """Parse a SubRip (.srt) subtitle file into a list of cues.

    Each cue is a tuple of (start_time, end_time, text_lines).  Times are in
    seconds.  Text lines preserve their order and may contain multiple lines.
    """
    cues: List[Tuple[float, float, List[str]]] = []
    if not os.path.exists(path):
        print(f"Subtitle file '{path}' does not exist. Continuing without subtitles.")
        return cues
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().splitlines()

    idx = 0
    while idx < len(content):
        # Skip empty lines and index numbers
        if not content[idx].strip():
            idx += 1
            continue
        # Optional: skip index line (an integer)
        if content[idx].strip().isdigit():
            idx += 1
        # Parse timing line
        if idx >= len(content):
            break
        timing = content[idx].strip()
        idx += 1
        if "-->" not in timing:
            continue
        start_str, end_str = [x.strip() for x in timing.split("-->")]
        try:
            start_time = parse_timecode(start_str)
            end_time = parse_timecode(end_str)
        except ValueError:
            # Skip malformed cue
            continue
        # Collect subtitle lines until a blank line
        lines: List[str] = []
        while idx < len(content) and content[idx].strip():
            lines.append(content[idx].rstrip('\r'))
            idx += 1
        cues.append((start_time, end_time, lines))
        # Skip the blank line separating cues
        while idx < len(content) and not content[idx].strip():
            idx += 1
    return cues


def find_active_cue(cues: List[Tuple[float, float, List[str]]], current_time: float) -> Optional[List[str]]:
    """Return the text lines for the active subtitle at the given time, or None."""
    # Linear search; for efficiency a binary search could be used if cues were sorted
    for start, end, lines in cues:
        if start <= current_time <= end:
            return lines
    return None


def overlay_subtitle(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    """Overlay subtitle lines onto a copy of the given frame and return it.

    The subtitle is drawn at the bottom of the frame.  A semi‑transparent
    rectangle is drawn behind the text to improve readability.  The original
    frame is not modified.
    """
    # Copy the frame to avoid modifying the original
    result = frame.copy()
    h, w = result.shape[:2]
    if not lines:
        return result

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6  # relative size; adjust as needed
    thickness = 2
    line_spacing = 10  # pixels between lines

    # Determine size of each line
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(size[0] for size in text_sizes)
    total_text_height = sum(size[1] for size in text_sizes) + (len(lines) - 1) * line_spacing

    # Calculate position: bottom center
    margin_bottom = 20  # distance from bottom of frame
    y_start = h - margin_bottom - total_text_height
    x_start = (w - max_width) // 2

    # Draw semi‑transparent rectangle behind the text
    # Determine rectangle corners
    rect_x1 = x_start - 10
    rect_y1 = y_start - 5
    rect_x2 = x_start + max_width + 10
    rect_y2 = y_start + total_text_height + 5
    overlay = result.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    # Draw each line
    y = y_start
    for line, size in zip(lines, text_sizes):
        line_width, line_height = size
        x = (w - line_width) // 2
        cv2.putText(result, line, (x, y + line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height + line_spacing
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an MKV video and capture frames without subtitles.")
    parser.add_argument("--video", required=True, help="Path to the MKV video file to play")
    parser.add_argument("--subtitle", help="Path to a .srt subtitle file to display (optional)")
    parser.add_argument("--output-dir", default="screenshots", help="Directory to save screenshots")
    args = parser.parse_args()

    video_path = args.video
    subtitle_path = args.subtitle
    output_dir = args.output_dir

    if not os.path.exists(video_path):
        print(f"Error: video file '{video_path}' does not exist.")
        sys.exit(1)

    # If subtitle is not provided, try to use a .srt file with the same basename
    if not subtitle_path:
        base, _ = os.path.splitext(video_path)
        candidate = base + ".srt"
        if os.path.exists(candidate):
            subtitle_path = candidate
        else:
            subtitle_path = None

    cues: List[Tuple[float, float, List[str]]] = []
    if subtitle_path:
        cues = parse_srt(subtitle_path)
        if cues:
            print(f"Loaded {len(cues)} subtitles from {subtitle_path}.")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video file '{video_path}'.")
        sys.exit(1)

    # Get frames per second to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # default fallback fps
    delay = int(1000 / fps)

    paused = False
    current_frame: Optional[np.ndarray] = None
    window_name = "MKV Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Controls: press 'space' to pause/resume, 's' to take a screenshot, 'q' or Esc to quit.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            current_frame = frame
        # Use the last read frame if paused
        if current_frame is None:
            break
        display_frame = current_frame.copy()
        # Determine current playback time in seconds
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Find and overlay subtitle if available
        if cues:
            subtitle_lines = find_active_cue(cues, current_time)
            if subtitle_lines:
                display_frame = overlay_subtitle(display_frame, subtitle_lines)
        # Show frame
        cv2.imshow(window_name, display_frame)
        # Wait for key press and timing; waitKey returns key code or -1
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):
            # Quit on 'q' or Escape
            print("Quitting...")
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            # Save current_frame without subtitles
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, current_frame)
            print(f"Saved screenshot to {filepath}")
        # else ignore other keys

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()