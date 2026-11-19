"""Verify the frame-level spatial split of CottonBoll-UAV.

The authoritative split is splits/split_train.txt, splits/split_test.txt and
splits/split_buffer.txt (one frame stem per line). This script checks that:

  1. the per-field counts match the paper (Table 2):
       pre-defoliation   train 544 / test 154 / buffer 148
       post-defoliation  train 657 / test 182 / buffer 172
  2. no frame appears in more than one split;
  3. metadata/frames.csv agrees with the split files;
  4. within each field, every test frame is at least 13.0 m from every training
     frame along the flight axis (x_m in metadata/frames.csv; the axis definition
     is recorded in metadata/split_summary.txt).

Usage:  python verify_split.py            (run from the repository root)
Exit code 0 if every check passes, 1 otherwise.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "pre": {"train": 544, "test": 154, "buffer": 148},
    "post": {"train": 657, "test": 182, "buffer": 172},
}
MIN_SEPARATION_M = 13.0

failures = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        failures.append(msg)


# Split files
splits = {}
for name in ("train", "test", "buffer"):
    path = ROOT / "splits" / f"split_{name}.txt"
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(len(stems) == len(set(stems)), f"{path.name}: no duplicate lines ({len(stems)} frames)")
    splits[name] = set(stems)

# 2. No frame in two splits
for a, b in (("train", "test"), ("train", "buffer"), ("test", "buffer")):
    both = splits[a] & splits[b]
    check(not both, f"no frame in both {a} and {b}" + (f" (found {len(both)}, e.g. {sorted(both)[0]})" if both else ""))

# frames.csv
with open(ROOT / "metadata" / "frames.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
mapped = [r for r in rows if r["field"] in EXPECTED]
stem = lambda r: r["frame"].rsplit(".", 1)[0]

# 1. Counts per field
for field, exp in EXPECTED.items():
    stems = {stem(r) for r in mapped if r["field"] == field}
    got = {name: len(splits[name] & stems) for name in exp}
    check(got == exp, f"{field}-defoliation counts {got} == {exp}")
all_split = set().union(*splits.values())
check(all_split == {stem(r) for r in mapped},
      f"split files cover exactly the {len(mapped)} mapping frames in frames.csv")

# 3. frames.csv split column agrees with the split files
lookup = {s: name for name, ss in splits.items() for s in ss}
disagree = [r["frame"] for r in mapped if lookup.get(stem(r)) != r["split"]]
check(not disagree, "frames.csv split column agrees with splits/*.txt"
      + (f" ({len(disagree)} disagree, e.g. {disagree[0]})" if disagree else ""))
keys = Counter((r["field"], r["run"], r["sequence"]) for r in rows)
check(max(keys.values()) == 1, "(field, run, sequence) is unique in frames.csv")

# 4. Minimum along-track separation between test and train
for field in EXPECTED:
    xs = {name: sorted(float(r["x_m"]) for r in mapped if r["field"] == field and r["split"] == name)
          for name in ("train", "test")}
    sep = min(abs(t - x) for t in xs["test"] for x in xs["train"])
    check(sep >= MIN_SEPARATION_M,
          f"{field}-defoliation: min along-track test-train separation {sep:.2f} m >= {MIN_SEPARATION_M} m")

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
