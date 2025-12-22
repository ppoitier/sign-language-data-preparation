from tqdm import tqdm
import cv2
from vidgear.gears import CamGear


def iterate_video_frames_using_vidgear(video_path: str, show_progress=False):
    capture = CamGear(source=video_path).start()
    n_frames = int(capture.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.stream.get(cv2.CAP_PROP_FPS))
    progress_bar = tqdm(range(n_frames), disable=not show_progress)
    for frame_nb in progress_bar:
        timestamp_ms = int(round(frame_nb * 1000 / fps))
        frame = capture.read()
        if frame is None:
            progress_bar.write(f"Cannot read frame {frame_nb}.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield timestamp_ms, frame
    capture.stop()