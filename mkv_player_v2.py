#!/usr/bin/env python3
"""
Simple MKV video player with subtitle‑free screen capture.

This script uses the OpenCV library to open and play back a video file (such as
an MKV).  It focuses on the video track; internal soft subtitles in the MKV
container are not rendered because OpenCV’s `VideoCapture` API decodes only
the video stream.  When the user presses the **s** key during playback,
the program saves a screenshot of the current frame to disk.  Because no
subtitle rendering is performed, these screenshots naturally exclude any
embedded subtitles.  The filename of each screenshot includes a timestamp so
that multiple captures do not overwrite each other.  The viewer also
supports pausing and resuming playback with the space bar and exiting with
**q** or Escape.

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
with the `--video` option.  The internal video stream will be played; any
embedded subtitle tracks in the MKV will not appear in the window.  Screenshots
are stored in the directory specified by `--output-dir` (default `screenshots`).

Example:

    python mkv_player.py --video mymovie.mkv

Key bindings while the player window is open:

* **Space** – pause/resume playback.
* **s** – save a screenshot of the current frame (which contains only the raw video).
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
`destroyAllWindows()` and `cap.release()` free resources【5417716936850†L270-L274】.
When the user requests a screenshot, the program simply writes the raw
current frame to disk using `cv2.imwrite`; because no subtitles are rendered
by OpenCV, the captured image contains only the video【47430801850246†L185-L188】.
If you were using the MPV media player, pressing **S** takes a screenshot
without subtitles whereas **s** includes them【47430801850246†L185-L188】.  This script
achieves the same effect by never drawing subtitles on top of the video.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import cv2
import numpy as np




def main() -> None:
    parser = argparse.ArgumentParser(description="Play an MKV video and capture frames without subtitles.")
    parser.add_argument("--video", required=True, help="Path to the MKV video file to play")
    # No external subtitle support is used here.  OpenCV does not render
    # internal MKV subtitles, so the video is shown without text.
    parser.add_argument("--output-dir", default="screenshots", help="Directory to save screenshots")
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output_dir

    if not os.path.exists(video_path):
        print(f"Error: video file '{video_path}' does not exist.")
        sys.exit(1)
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
        # Nothing to overlay; OpenCV does not render internal subtitles.
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