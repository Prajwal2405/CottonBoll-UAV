"""Regenerate the CottonBoll-UAV tiles from the released full-resolution frames.

Each 5280 x 3956 frame is cut into 640 x 640 tiles with 100 px overlap (stride 540 px).
The last column / row is shifted inward to end exactly at the image edge, giving
10 columns x 8 rows = 80 tiles per frame:

    x (left)  = 0, 540, 1080, 1620, 2160, 2700, 3240, 3780, 4320, 4640
    y (top)   = 0, 540, 1080, 1620, 2160, 2700, 3240, 3316

Tile file name:  <frame stem>_r<row>_c<col>_x<left>_y<top>.jpg
    e.g. DJI_20250929091533_0001_D_r0_c9_x4640_y0.jpg
         = frame DJI_20250929091533_0001_D, row 0, column 9, top-left corner (4640, 0) px.
This is the naming used by the training labels (train_labels_yolo.zip) and the evaluation set.

Tiles are written with OpenCV at JPEG quality 95, which reproduces the tiles used in the
paper byte-for-byte.

Usage:
    python make_tiles.py --frames path/to/frames_pre_defoliation --out tiles/pre
    python make_tiles.py --frames path/to/frames --out tiles --split train   # training frames only
    python make_tiles.py --frames ... --out ... --per-frame-dirs            # one sub-folder per frame

--split uses splits/split_<name>.txt from this repository.
"""

import argparse
import sys
from pathlib import Path

import cv2

TILE = 640
OVERLAP = 100
STRIDE = TILE - OVERLAP          # 540
JPEG_QUALITY = 95


def starts(length: int) -> list[int]:
    """Tile origins along one axis: multiples of STRIDE, last one clamped to the edge."""
    out = list(range(0, length - TILE + 1, STRIDE))
    if out[-1] + TILE < length:
        out.append(length - TILE)
    return out


def tile_frame(frame_path: Path, out_dir: Path) -> int:
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read {frame_path}")
    h, w = img.shape[:2]
    stem = frame_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r, y in enumerate(starts(h)):
        for c, x in enumerate(starts(w)):
            name = f"{stem}_r{r}_c{c}_x{x}_y{y}.jpg"
            ok = cv2.imwrite(str(out_dir / name), img[y:y + TILE, x:x + TILE],
                             [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"could not write {out_dir / name}")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, type=Path, help="folder of full-resolution DJI_*.JPG frames (searched recursively)")
    ap.add_argument("--out", required=True, type=Path, help="output folder for tiles")
    ap.add_argument("--split", choices=["train", "test", "buffer"], help="only tile frames listed in splits/split_<name>.txt")
    ap.add_argument("--per-frame-dirs", action="store_true", help="write each frame's tiles to <out>/<frame stem>/")
    args = ap.parse_args()

    frames = sorted(p for p in args.frames.rglob("*") if p.suffix.lower() == ".jpg" and p.name.upper().startswith("DJI_"))
    if args.split:
        split_file = Path(__file__).resolve().parent / "splits" / f"split_{args.split}.txt"
        keep = {s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()}
        frames = [p for p in frames if p.stem in keep]
    if not frames:
        print("no frames found", file=sys.stderr)
        return 1

    total = 0
    for i, f in enumerate(frames, 1):
        total += tile_frame(f, args.out / f.stem if args.per_frame_dirs else args.out)
        if i % 50 == 0 or i == len(frames):
            print(f"{i}/{len(frames)} frames, {total} tiles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
