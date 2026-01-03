#!/usr/bin/env python3
from __future__ import annotations

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

# pydicom LUT helpers (버전에 따라 위치가 다를 수 있어서 try/except)
try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except Exception:
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# -------------------------
# DaTScan DICOM helpers
# -------------------------
def _ensure_zyx(arr: np.ndarray, n_frames: int | None) -> np.ndarray:
    """
    DICOM pixel_array의 축을 (Z,H,W)로 최대한 안정적으로 맞춤.
    """
    if arr.ndim == 2:
        return arr[None, ...]  # (1,H,W)

    if arr.ndim != 3:
        # 예상 밖이면 마지막 두 축을 H,W로 보고 앞을 프레임으로 강제
        arr = np.reshape(arr, (-1, arr.shape[-2], arr.shape[-1]))
        return arr

    sh = arr.shape

    # NumberOfFrames가 있으면 일치하는 축을 Z로
    if n_frames is not None:
        for ax, L in enumerate(sh):
            if L == n_frames:
                if ax == 0:
                    return arr
                if ax == 1:
                    return np.transpose(arr, (1, 0, 2))
                if ax == 2:
                    return np.transpose(arr, (2, 0, 1))

    # 휴리스틱: 가장 작은 축을 Z로 가정
    axes = np.argsort(sh)  # ascending
    z_ax = int(axes[0])
    if z_ax == 0:
        return arr
    if z_ax == 1:
        return np.transpose(arr, (1, 0, 2))
    return np.transpose(arr, (2, 0, 1))


def load_datscan_dcm(path: Path) -> np.ndarray:
    """
    - pixel_array 로드
    - Modality LUT/Rescale 적용 (가능하면)
    - VOI LUT(Windowing) 적용 (가능하면)
    - MONOCHROME1이면 반전
    - 최종 (Z,H,W) float32 반환
    """
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    n_frames = int(getattr(ds, "NumberOfFrames", 0)) or None

    arr = _ensure_zyx(arr, n_frames=n_frames).astype(np.float32)

    out = np.empty_like(arr, dtype=np.float32)
    for z in range(arr.shape[0]):
        sl = arr[z]

        # modality LUT/rescale -> VOI LUT(windowing)
        try:
            sl = apply_modality_lut(sl, ds)
        except Exception:
            pass
        try:
            sl = apply_voi_lut(sl, ds)
        except Exception:
            pass

        sl = np.asarray(sl, dtype=np.float32)

        # MONOCHROME1: invert
        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            sl = sl.max() - sl

        out[z] = sl

    return out  # (Z,H,W)


def normalize_datscan_slice(
    x: np.ndarray,
    bg_p: float = 15,
    p_low: float = 5,
    p_high: float = 99,
    gamma: float = 0.7,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    DaTScan 2D slice 정규화:
    - 약한 배경 제거(bg percentile)
    - percentile clip
    - 0~1 스케일
    - gamma로 hotspot 가시성 확보
    """
    x = x.astype(np.float32)

    bg = np.percentile(x, bg_p)
    x = np.clip(x - bg, 0, None)

    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    x = np.clip(x, 0, 1) ** gamma
    return x


def pick_datscan_striatum_slices(dat_vol: np.ndarray, k: int = 7) -> list[int]:
    """
    선조체(핫스팟)가 있는 중심 슬라이스를 고르되,
    핫스팟 중심이 너무 가장자리면 패널티를 줘서 '끝에 붙는' 이상 프레임을 회피.
    """
    Z, H, W = dat_vol.shape
    scores: list[float] = []

    for z in range(Z):
        sl = dat_vol[z]
        thr = np.percentile(sl, 95)
        m = sl >= thr

        if m.sum() < 10:
            scores.append(0.0)
            continue

        ys, xs = np.where(m)
        cy, cx = float(ys.mean()), float(xs.mean())

        margin = 0.15
        edge = (cy < margin * H) or (cy > (1 - margin) * H) or (cx < margin * W) or (cx > (1 - margin) * W)
        edge_penalty = 0.5 if edge else 0.0

        score = float(sl[m].sum() * (1.0 - edge_penalty))
        scores.append(score)

    center = int(np.argmax(scores))
    half = k // 2
    start = max(0, center - half)
    end = min(Z, center + half + 1)
    return list(range(start, end))


# -------------------------
# Generic helpers
# -------------------------
def find_ppmi_root(root: Path) -> Path:
    """
    사용자가 /workspace/PPMI 라고 했지만, 트리상 PPMI/PPMI/<SUBJECT>/... 구조가 흔해서 자동 보정.
    """
    root = root.resolve()
    if (root / "PPMI" / "PPMI").is_dir():
        return root / "PPMI" / "PPMI"
    if (root / "PPMI").is_dir():
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
    nii = nib.as_closest_canonical(nii)
    vol = nii.get_fdata().astype(np.float32)
    return vol


def resize_square(rgb: np.ndarray, resolution: int, resample=Image.BICUBIC) -> Image.Image:
    im = Image.fromarray(rgb)
    return im.resize((resolution, resolution), resample=resample)


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


def export_dataset(pairs, out_root, resolution, n_slices, dat_crop=0, dat_crop_mode="hotspot", dat_crop_margin=0.15):
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
                t1_vol = load_t1_nii(t1_path)        # (X,Y,Z)
                dat_vol = load_datscan_dcm(dat_path) # (Z,H,W) display-ready
            except Exception as e:
                print(f"[WARN] load failed: {subject} :: {e}")
                continue

            # T1 normalize (볼륨 단위: 일단 유지. 필요하면 slice-wise로 바꿔도 됨)
            t1_vol01 = robust_minmax01(t1_vol)
            z_t1 = t1_vol01.shape[-1]

            # DaTScan slice 선택
            dat_idxs = pick_datscan_striatum_slices(dat_vol, k=n_slices)

            for k, z in enumerate(dat_idxs):
                # ----- DaT -----
                dat_sl = dat_vol[z]
                dat_sl01 = normalize_datscan_slice(dat_sl)

                if float(dat_sl01.max()) < 1e-3:
                    continue

                # --- ROI crop (옵션) ---
                if dat_crop and dat_crop > 0:
                    if dat_crop_mode == "center":
                        dat_sl01 = crop_square_center(dat_sl01, dat_crop)
                    else:
                        dat_sl01 = crop_square_hotspot(dat_sl01, dat_crop, margin=dat_crop_margin)

                dat_img = resize_square(to_uint8_rgb(dat_sl01), resolution, resample=Image.BILINEAR)

                # ----- T1 (DaT z에 대응시키는 단순 비율 매핑) -----
                z_t1_use = min(int(round(z / max(1, (dat_vol.shape[0] - 1)) * (z_t1 - 1))), z_t1 - 1)
                t1_sl = t1_vol01[..., z_t1_use]
                t1_img = resize_square(to_uint8_rgb(t1_sl), resolution, resample=Image.BICUBIC)

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

def crop_square_center(img01: np.ndarray, crop: int) -> np.ndarray:
    H, W = img01.shape
    crop = int(crop)
    crop = min(crop, H, W)
    cy, cx = H // 2, W // 2
    y0 = max(0, cy - crop // 2)
    x0 = max(0, cx - crop // 2)
    y1 = min(H, y0 + crop)
    x1 = min(W, x0 + crop)
    return img01[y0:y1, x0:x1]


def crop_square_hotspot(img01: np.ndarray, crop: int, margin: float = 0.15) -> np.ndarray:
    """
    상위 intensity 픽셀들의 중심(centroid)을 기준으로 square crop.
    핫스팟이 가장자리에 붙으면(축 꼬임/이상 프레임) center crop로 폴백.
    """
    H, W = img01.shape
    crop = int(crop)
    crop = min(crop, H, W)

    thr = np.percentile(img01, 95)
    m = img01 >= thr

    if m.sum() < 10:
        return crop_square_center(img01, crop)

    ys, xs = np.where(m)
    cy, cx = float(ys.mean()), float(xs.mean())

    # 가장자리면 폴백
    if (cy < margin * H) or (cy > (1 - margin) * H) or (cx < margin * W) or (cx > (1 - margin) * W):
        return crop_square_center(img01, crop)

    y0 = int(round(cy - crop / 2))
    x0 = int(round(cx - crop / 2))
    y0 = max(0, min(y0, H - crop))
    x0 = max(0, min(x0, W - crop))
    return img01[y0:y0 + crop, x0:x0 + crop]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ppmi_root", type=str, default="/workspace/PPMI")
    ap.add_argument("--out_root", type=str, default="/workspace/mri2datscan/ppmi_mri2datscan")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--n_slices", type=int, default=8)
    ap.add_argument("--max_delta_days", type=int, default=180)
    ap.add_argument("--max_pairs", type=int, default=0, help="0이면 전체 사용")
    ap.add_argument("--dat_crop", type=int, default=0, help="DaTScan ROI crop size (0이면 crop 안함). 예: 160")
    ap.add_argument("--dat_crop_mode", type=str, default="hotspot", choices=["center", "hotspot"],
                    help="center: 중앙 크롭, hotspot: 핫스팟 중심 기반 크롭")
    ap.add_argument("--dat_crop_margin", type=float, default=0.15, help="hotspot 크롭 시 가장자리 제외 마진(0~0.5)")

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

    export_dataset(
        pairs,
        out_root=out_root,
        resolution=args.resolution,
        n_slices=args.n_slices,
        dat_crop=args.dat_crop,
        dat_crop_mode=args.dat_crop_mode,
        dat_crop_margin=args.dat_crop_margin,
    )


if __name__ == "__main__":
    main()
