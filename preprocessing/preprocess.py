# Purpose: read videos, find the face in a few frames of each video, crop it, and save it as a 224x224 image.

# Libraries we need:
# argparse = read options from the command line, csv = write the log file, cv2 = read video and save images,
# numpy = number tools, pandas = read the splits table, torch = check if a GPU exists
import argparse, csv, cv2, numpy as np, pandas as pd, torch
from pathlib import Path          # Path = easy way to build file and folder paths
from tqdm import tqdm             # tqdm = shows a progress bar
from facenet_pytorch import MTCNN # MTCNN = a model that finds faces in an image


# Cut a square area around the face and resize it to size x size.
def crop_face(frame, box, margin=1.3, size=224):
    x1, y1, x2, y2 = box                          # corners of the face box: left, top, right, bottom
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2         # centre point of the face box
    half = max(x2 - x1, y2 - y1) * margin / 2     # half of the square side: longer side of box, made 1.3x bigger
    h, w = frame.shape[:2]                        # height and width of the full frame
    # Make the square around the centre, but keep it inside the frame (no negative or too-large values)
    x1, x2 = int(max(cx - half, 0)), int(min(cx + half, w))
    y1, y2 = int(max(cy - half, 0)), int(min(cy + half, h))
    face = frame[y1:y2, x1:x2]                    # cut out that part of the frame
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)  # resize to 224x224 (INTER_AREA is good for shrinking)


# Main work happens here. "a" holds the command-line options.
def main(a):
    df = pd.read_csv(a.splits)                    # load the splits table (one row per video)
    # FaceShifter is used only for testing, so drop its train/val rows
    df = df[~((df.method == "FaceShifter") & (df.split != "test"))]
    if a.methods:                                 # if the user listed specific methods...
        df = df[df.method.isin(a.methods)]        # ...keep only those methods
    # Limit train videos, but keep all val/test videos
    if a.max_train:
        tr = df[df.split == "train"].groupby("method").head(a.max_train)  # first N train videos of each method
        df = pd.concat([tr, df[df.split != "train"]])                     # join limited train + full val/test
    if a.max_per_group:                           # small test run: first N videos of each (method, split) group
        df = df.groupby(["method", "split"]).head(a.max_per_group)

    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)  # output folder (create if missing)
    log_path = out_root / "frames.csv"            # log file that lists every saved face
    done = set()                                  # videos already processed
    if log_path.exists():                         # resume: if a log exists, skip videos already done
        prev = pd.read_csv(log_path)
        done = set(zip(prev.split, prev.method, prev.video_id))
    is_new = not log_path.exists()                # True if this is a fresh run (no log yet)
    f = open(log_path, "a", newline="")           # open the log in append mode so old rows are kept
    w = csv.writer(f)                             # helper to write rows into the log
    if is_new:
        w.writerow(["split", "method", "label", "video_id", "path"])  # write the header only once

    # Create the face detector: pick the largest face if many; use GPU if available, otherwise CPU
    mtcnn = MTCNN(select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu")
    n_saved = 0                                   # counter for faces saved in this run
    for r in tqdm(df.itertuples(), total=len(df)):  # go through each video row with a progress bar
        if (r.split, r.method, r.video_id) in done:  # already processed earlier?
            continue                              # then skip it
        cap = cv2.VideoCapture(str(Path(a.videos) / r.rel_path))  # open the video file
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames in the video
        if n <= 0:                                # 0 frames means the video could not be opened
            print("could not open:", Path(a.videos) / r.rel_path)
            continue
        out_dir = out_root / r.split / r.method / r.video_id  # folder for this video's faces
        # Pick a.frames frame numbers spread evenly from first to last frame
        for k, idx in enumerate(np.linspace(0, n - 1, a.frames).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))  # jump to that frame
            ok, frame = cap.read()                # read the frame (ok = False if reading failed)
            if not ok:
                continue                          # could not read it, try the next frame
            # Find faces; OpenCV gives BGR colours but MTCNN expects RGB, so convert first
            boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if boxes is None:
                continue                          # no face found, skip this frame
            face = crop_face(frame, boxes[0], a.margin)  # crop the first (largest) face
            out_dir.mkdir(parents=True, exist_ok=True)   # make the video's folder if needed
            p = out_dir / f"{k:02d}.jpg"          # file name like 00.jpg, 01.jpg, ...
            cv2.imwrite(str(p), face, [cv2.IMWRITE_JPEG_QUALITY, 95])  # save as JPEG with high quality
            # Add a log row: split, method, label, video id, and image path relative to the output folder
            w.writerow([r.split, r.method, r.label, r.video_id, str(p.relative_to(out_root))])
            n_saved += 1                          # count this saved face
        cap.release()                             # close the video file
        f.flush()                                 # write the log to disk after every video (safe if the run stops)
    f.close()                                     # close the log file
    print("saved", n_saved, "new faces")          # final summary


# This block runs only when you start the file directly (not when imported)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()                # tool to read command-line options
    ap.add_argument("--videos", default="/kaggle/input/datasets/xdxd003")   # folder with the raw videos
    ap.add_argument("--splits", default="/kaggle/working/splits.csv")       # path of the splits file
    ap.add_argument("--out", default="/kaggle/working/faces")               # where to save faces
    ap.add_argument("--frames", type=int, default=10)       # frames to take from each video
    ap.add_argument("--margin", type=float, default=1.3)    # how much bigger than the face box to crop
    ap.add_argument("--methods", nargs="*")                 # optional list of methods to process
    ap.add_argument("--max-train", type=int, default=0)     # max train videos per method (0 = no limit)
    ap.add_argument("--max-per-group", type=int, default=0) # max videos per method+split, for quick tests (0 = no limit)
    main(ap.parse_args())                         # read the options and start
