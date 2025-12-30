#!/usr/bin/env python3
import re
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import nibabel as nib
import pydicom


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def find_ppmi_root(root: Path) -> Path:
    """
    사용자가 /workspace/PPMI 라고 했지만, 트리상 PPMI/PPMI/<SUBJECT>/... 구조가 흔해서 자동 보정.
    """
    root = root.resolve()
    if (root / "PPMI" / "PPMI").is_dir():
        return root / "PPMI" / "PPMI"
    if (root / "PPMI").is_dir():
        # 어떤 덤프는 /workspace/PPMI/PPMI/<SUBJECT> 형태
        # root/PPMI 아래에 숫자 subject 폴더가 없고 PPMI가 또 있으면 한 번 더 들어감
        if (root / "PPMI" / "PPMI").is_dir():
            return root / "PPMI" / "PPMI"
        return root / "PPMI"
    return root


def extract_date_from_path(p: Path):
    s = "/".join(p.parts)
    m = DATE_RE.search(s)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def list_modality_files(subject_dir: Path, modality_dirname: str, exts: tuple[str, ...]) -> list[Path]:
    mod = subject_dir / modality_dirname
    if not mod.is_dir():
        return []
    files = []
    for ext in exts:
        files.extend(mod.rglob(f"*{ext}"))
    # xml은 제외(같은 폴더에 xml만 있는 케이스가 많음)
    files = [p for p in files if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def robust_minmax01(x: np.ndarray, p_low=0.5, p_high=99.5, eps=1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)


def to_uint8_rgb(img01_2d: np.ndarray) -> np.ndarray:
    u8 = (np.clip(img01_2d, 0, 1) * 255.0).round().astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def load_t1_nii(path: Path) -> np.ndarray:
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)  # 방향 통일(최대한)
    vol = nii.get_fdata().astype(np.float32)
    return vol


def load_datscan_dcm(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    # arr shape이 (H,W)면 1-slice로 취급
    if arr.ndim == 2:
        arr = arr[None, ...]  # (Z,H,W)
    # arr shape이 (Z,H,W)라고 가정 (대부분 multi-frame은 이렇게 나옴)
    # 만약 (H,W,Z)인 케이스는 Z가 너무 작거나 큰지 보고 휴리스틱으로 전치
    if arr.shape[0] < 8 and arr.shape[-1] >= 16:
        # Z가 뒤에 붙은 형태 의심
        arr = np.transpose(arr, (2, 0, 1))
    return arr  # (Z,H,W)


def pick_slice_indices(z: int, n_slices: int, frac_low=0.35, frac_high=0.65) -> list[int]:
    if z <= 1:
        return [0]
    a = int(math.floor(frac_low * z))
    b = int(math.ceil(frac_high * z)) - 1
    a = max(0, min(a, z - 1))
    b = max(0, min(b, z - 1))
    if b <= a:
        a, b = 0, z - 1
    if n_slices == 1:
        return [(a + b) // 2]
    return np.linspace(a, b, n_slices).round().astype(int).tolist()


def resize_square(rgb: np.ndarray, resolution: int) -> Image.Image:
    im = Image.fromarray(rgb)
    return im.resize((resolution, resolution), resample=Image.BICUBIC)


@dataclass
class PairRow:
    subject: str
    t1_path: str
    dat_path: str
    t1_date: str
    dat_date: str
    delta_days: int


def build_pairs(ppmi_root: Path, max_delta_days: int) -> pd.DataFrame:
    rows: list[PairRow] = []
    subjects = sorted([p for p in ppmi_root.iterdir() if p.is_dir() and p.name.isdigit()])

    for sdir in tqdm(subjects, desc="Index subjects"):
        subject = sdir.name

        t1_files = list_modality_files(sdir, "T1-anatomical", (".nii", ".nii.gz"))
        dat_files = list_modality_files(sdir, "Reconstructed_DaTSCAN", (".dcm",))

        if not t1_files or not dat_files:
            continue

        t1 = []
        for p in t1_files:
            d = extract_date_from_path(p)
            if d:
                t1.append((d, p))
        dat = []
        for p in dat_files:
            d = extract_date_from_path(p)
            if d:
                dat.append((d, p))

        if not t1 or not dat:
            continue

        # 각 DaTScan에 대해 가장 가까운 T1을 매칭
        for d_dat, p_dat in dat:
            best = None
            for d_t1, p_t1 in t1:
                delta = abs((d_dat - d_t1).days)
                if best is None or delta < best[0]:
                    best = (delta, d_t1, p_t1)
            if best is None:
                continue
            delta, d_t1, p_t1 = best
            if delta <= max_delta_days:
                rows.append(
                    PairRow(
                        subject=subject,
                        t1_path=str(p_t1),
                        dat_path=str(p_dat),
                        t1_date=str(d_t1),
                        dat_date=str(d_dat),
                        delta_days=int(delta),
                    )
                )

    return pd.DataFrame([r.__dict__ for r in rows])


def export_dataset(pairs: pd.DataFrame, out_root: Path, resolution: int, n_slices: int):
    """
    out_root/
      data/train/targets/*.png
      data/train/conditioning/*.png
      data/train/metadata.jsonl   (for reference / debugging)
    """
    targets_dir = out_root / "data" / "train" / "targets"
    cond_dir = out_root / "data" / "train" / "conditioning"
    targets_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "data" / "train" / "metadata.jsonl"
    prompt = "brain DaTSCAN SPECT, grayscale"

    n_written = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Export pairs"):
            subject = row["subject"]
            t1_path = Path(row["t1_path"])
            dat_path = Path(row["dat_path"])

            try:
                t1_vol = load_t1_nii(t1_path)               # (X,Y,Z)
                dat_vol = load_datscan_dcm(dat_path)        # (Z,H,W)
            except Exception as e:
                print(f"[WARN] load failed: {subject} :: {e}")
                continue

            # normalize
            t1_vol01 = robust_minmax01(t1_vol)
            dat_vol01 = robust_minmax01(dat_vol)

            # slice indices (T1: axis=-1, DaT: axis=0)
            z_t1 = t1_vol01.shape[-1]
            z_dat = dat_vol01.shape[0]
            idxs = pick_slice_indices(min(z_t1, z_dat), n_slices=n_slices)

            for k, z in enumerate(idxs):
                # T1 slice: (X,Y)
                t1_sl = t1_vol01[..., min(z, z_t1 - 1)]
                # DaT slice: (H,W)
                dat_sl = dat_vol01[min(z, z_dat - 1), ...]

                # 크기 불일치면 DaT를 T1 크기에 맞춰 리사이즈(학습용 단순화)
                # (정밀 registration은 나중에 추가 권장)
                t1_rgb = to_uint8_rgb(t1_sl)
                dat_rgb = to_uint8_rgb(dat_sl)

                t1_img = resize_square(t1_rgb, resolution)
                dat_img = resize_square(dat_rgb, resolution)

                stem = f"{subject}_dt{row['dat_date']}_t1{row['t1_date']}_z{z:03d}_{k:02d}"
                rel_target = f"data/train/targets/{stem}.png"
                rel_cond = f"data/train/conditioning/{stem}.png"

                dat_img.save(out_root / rel_target)
                t1_img.save(out_root / rel_cond)

                rec = {
                    "image": rel_target,
                    "conditioning_image": rel_cond,
                    "text": prompt,
                    "subject": subject,
                    "t1_date": row["t1_date"],
                    "dat_date": row["dat_date"],
                    "delta_days": int(row["delta_days"]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Done. wrote {n_written} slice-pairs")
    print(f"metadata: {meta_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppmi_root", type=str, default="/workspace/PPMI")
    ap.add_argument("--out_root", type=str, default="/workspace/mri2datscan/ppmi_mri2datscan")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--n_slices", type=int, default=8)
    ap.add_argument("--max_delta_days", type=int, default=180)
    ap.add_argument("--max_pairs", type=int, default=0, help="0이면 전체 사용")
    args = ap.parse_args()

    ppmi_root = find_ppmi_root(Path(args.ppmi_root))
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(ppmi_root, max_delta_days=args.max_delta_days)
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs.sample(args.max_pairs, random_state=42).reset_index(drop=True)

    pairs_path = out_root / "pairs_manifest.csv"
    pairs.to_csv(pairs_path, index=False)
    print(f"pairs: {len(pairs)}  | manifest: {pairs_path}")

    export_dataset(pairs, out_root=out_root, resolution=args.resolution, n_slices=args.n_slices)


if __name__ == "__main__":
    main()
