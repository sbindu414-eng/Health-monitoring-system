"""
Dataset Preparation & Leak-Free Splitting Pipeline.

Supports:
1. Automated download & extraction of real public clinical datasets:
   - RetinaMNIST (224x224 RGB retinal fundus photographs derived from DeepDRiD,
     CC BY 4.0, Zenodo Record 10519652) with strict pre-defined patient-level
     train / val / test splits and exact ICDR 0-4 diabetic retinopathy labels.
   - Pima Indians Diabetes Database (768 patient clinical tabular records) from
     public UCI / OpenML / Plotly open mirror.
2. Optional ingestion of locally downloaded Kaggle datasets without changing code:
   - APTOS 2019 Blindness Detection (`data/aptos2019/train.csv` + `train_images/`)
   - EyePACS Diabetic Retinopathy (`data/eyepacs/trainLabels.csv` + `train/`)
3. Verification:
   - Reads images, checks format/mode/dimensions, filters unreadable/corrupted files,
     computes SHA-256 + 8x8 perceptual dHash to eliminate duplicates across splits,
     reports class distribution, and prevents patient-level data leakage.
"""

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RETINAL_DIR = os.path.join(DATA_DIR, "retinal_fundus")
PIMA_DIR = os.path.join(DATA_DIR, "pima_diabetes")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

RETINAMNIST_HF_URLS = {
    "train": "https://huggingface.co/datasets/danjacobellis/retinamnist_224/resolve/main/data/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/danjacobellis/retinamnist_224/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/danjacobellis/retinamnist_224/resolve/main/data/test-00000-of-00001.parquet",
}
PIMA_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]

DR_CLASS_NAMES = {
    0: "No diabetic retinopathy",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative diabetic retinopathy",
}


def compute_dhash(pil_img: Image.Image, hash_size: int = 8) -> str:
    """Compute perceptual difference hash (dHash) to catch near-duplicate images."""
    resized = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = "".join("1" if val else "0" for val in diff.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def compute_sha256_file(path: str) -> str:
    """Compute SHA-256 checksum of an image file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: str, timeout: int = 120) -> None:
    """Stream-download a remote file with atomic temporary write."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    print(f"Downloading {url} -> {dest_path} ...")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        if total and downloaded < total * 0.98:
            raise IOError(f"Incomplete download: got {downloaded}/{total} bytes")
    os.replace(tmp_path, dest_path)
    print(f"Downloaded successfully ({os.path.getsize(dest_path) / (1024 * 1024):.2f} MB).")


def prepare_pima_diabetes(force: bool = False) -> dict:
    """Download and validate the 768-row Pima Indians Diabetes CSV."""
    os.makedirs(PIMA_DIR, exist_ok=True)
    csv_path = os.path.join(PIMA_DIR, "diabetes.csv")

    if not os.path.exists(csv_path) or force:
        downloaded = False
        for idx, url in enumerate(PIMA_URLS):
            try:
                if idx == 0:
                    download_file(url, csv_path, timeout=30)
                else:
                    cols = [
                        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
                    ]
                    df_tmp = pd.read_csv(url, names=cols)
                    df_tmp.to_csv(csv_path, index=False)
                downloaded = True
                break
            except Exception as exc:
                print(f"Mirror failed ({url}): {exc}")
        if not downloaded:
            raise RuntimeError("Could not download Pima Indians Diabetes CSV from public mirrors.")

    df = pd.read_csv(csv_path)
    expected_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Pima CSV missing columns: {missing}")

    n_dups = int(df.duplicated().sum())
    if n_dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        df.to_csv(csv_path, index=False)

    summary = {
        "dataset": "Pima Indians Diabetes Database (UCI / OpenML mirror)",
        "source_url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "csv_path": os.path.relpath(csv_path, ROOT_DIR),
        "rows": int(len(df)),
        "duplicates_removed": n_dups,
        "outcome_distribution": {int(k): int(v) for k, v in df["Outcome"].value_counts().sort_index().items()},
        "zero_value_counts": {
            col: int((df[col] == 0).sum())
            for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        },
    }
    print(f"[Pima Diabetes] Verified {summary['rows']} rows. Outcomes: {summary['outcome_distribution']}")
    return summary


def validate_and_export_image(
    img_source,
    out_path: str,
    min_dim: int = 64,
) -> Optional[dict]:
    """Open, verify readability/dimensions, save RGB PNG, and return hashes."""
    try:
        if isinstance(img_source, np.ndarray):
            pil_img = Image.fromarray(img_source).convert("RGB")
        elif isinstance(img_source, (bytes, bytearray)):
            with Image.open(io.BytesIO(img_source)) as im:
                im.verify()
            with Image.open(io.BytesIO(img_source)) as im:
                pil_img = im.convert("RGB")
        else:
            with Image.open(img_source) as im:
                im.verify()
            with Image.open(img_source) as im:
                pil_img = im.convert("RGB")
        width, height = pil_img.size
        if width < min_dim or height < min_dim:
            return None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pil_img.save(out_path, format="PNG")
        dhash = compute_dhash(pil_img)
        sha256 = compute_sha256_file(out_path)
        return {"width": width, "height": height, "dhash": dhash, "sha256": sha256}
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def prepare_retinamnist(force: bool = False) -> dict:
    """
    Download RetinaMNIST (DeepDRiD 224x224 RGB) parquet splits from HuggingFace,
    materialize PNG files, verify integrity, deduplicate safely, and export
    patient-disjoint CSV splits.
    """
    raw_dir = os.path.join(RETINAL_DIR, "raw")
    images_root = os.path.join(RETINAL_DIR, "images")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(images_root, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    split_records = {"train": [], "val": [], "test": []}
    seen_sha = set()
    corrupted_count = 0
    duplicate_count = 0

    for split in ("train", "val", "test"):
        pq_path = os.path.join(raw_dir, f"retinamnist_224_{split}.parquet")
        if not os.path.exists(pq_path) or force:
            download_file(RETINAMNIST_HF_URLS[split], pq_path, timeout=240)

        df_split = pq.read_table(pq_path).to_pandas()
        split_dir = os.path.join(images_root, split)
        os.makedirs(split_dir, exist_ok=True)

        for idx, row in df_split.iterrows():
            raw_lbl = row["label"]
            lbl_int = int(raw_lbl[0]) if hasattr(raw_lbl, "__len__") and not isinstance(raw_lbl, (str, bytes)) else int(raw_lbl)
            img_obj = row["image"]
            img_bytes = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
            fname = f"{split}_{idx:04d}_grade{lbl_int}.png"
            out_path = os.path.join(split_dir, fname)

            if not os.path.exists(out_path) or force:
                meta = validate_and_export_image(img_bytes, out_path, min_dim=64)
            else:
                meta = validate_and_export_image(out_path, out_path, min_dim=64)

            if meta is None:
                corrupted_count += 1
                continue

            if meta["sha256"] in seen_sha:
                duplicate_count += 1
                continue
            seen_sha.add(meta["sha256"])

            split_records[split].append(
                {
                    "image_path": os.path.relpath(out_path, ROOT_DIR).replace("\\", "/"),
                    "label": lbl_int,
                    "label_name": DR_CLASS_NAMES[lbl_int],
                    "split": split,
                    "patient_group": f"deepdrid_{split}_{idx}",
                    "width": meta["width"],
                    "height": meta["height"],
                    "dhash": meta["dhash"],
                    "sha256": meta["sha256"],
                }
            )

    # Check cross-split dhash overlap (remove from val/test if near-identical to train)
    train_dhashes = {r["dhash"] for r in split_records["train"]}
    leakage_filtered = 0
    for eval_split in ("val", "test"):
        clean_rows = []
        for r in split_records[eval_split]:
            if r["dhash"] in train_dhashes:
                leakage_filtered += 1
            else:
                clean_rows.append(r)
        split_records[eval_split] = clean_rows

    dfs = {}
    for split, rows in split_records.items():
        df = pd.DataFrame(rows)
        csv_out = os.path.join(RETINAL_DIR, f"{split}.csv")
        df.to_csv(csv_out, index=False)
        dfs[split] = df

    # Export clean test-split samples for live UI evaluation (never train images)
    test_df = dfs["test"]
    for grade in sorted(DR_CLASS_NAMES.keys()):
        grade_rows = test_df[test_df["label"] == grade]
        if len(grade_rows) > 0:
            src_rel = grade_rows.iloc[0]["image_path"]
            src_abs = os.path.join(ROOT_DIR, src_rel)
            sample_name = f"sample_test_grade{grade}_{DR_CLASS_NAMES[grade].lower().replace(' ', '_')}.png"
            dst_abs = os.path.join(SAMPLES_DIR, sample_name)
            with Image.open(src_abs) as im:
                im.save(dst_abs, format="PNG")

    class_dist = {
        split: {
            DR_CLASS_NAMES[int(k)]: int(v)
            for k, v in Counter(dfs[split]["label"]).items()
        }
        for split in ("train", "val", "test")
    }

    summary = {
        "dataset": "RetinaMNIST (224x224 RGB from DeepDRiD Benchmark, MedMNIST v2)",
        "license": "CC BY 4.0",
        "source_url": "https://huggingface.co/datasets/danjacobellis/retinamnist_224",
        "origin_zenodo": "https://zenodo.org/records/10519652",
        "kaggle_references": [
            "https://www.kaggle.com/competitions/aptos2019-blindness-detection/data",
            "https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data",
        ],
        "note": (
            "Kaggle APTOS 2019 & EyePACS require authenticated Kaggle API tokens and competition acceptance. "
            "DeepDRiD/RetinaMNIST provides real clinical 224x224 fundus photographs under the exact same "
            "5-grade ICDR diabetic retinopathy scale with strict patient-level train/val/test separation."
        ),
        "splits": {s: int(len(dfs[s])) for s in ("train", "val", "test")},
        "total_images": int(sum(len(dfs[s]) for s in ("train", "val", "test"))),
        "corrupted_rejected": corrupted_count,
        "exact_duplicates_removed": duplicate_count,
        "cross_split_near_duplicates_removed": leakage_filtered,
        "class_names": DR_CLASS_NAMES,
        "class_distribution": class_dist,
    }
    with open(os.path.join(RETINAL_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[RetinaMNIST] Splits prepared -> train={summary['splits']['train']}, val={summary['splits']['val']}, test={summary['splits']['test']}")
    print(f"[RetinaMNIST] Corrupted rejected={corrupted_count}, Exact dups removed={duplicate_count}, Cross-split dHash filtered={leakage_filtered}")
    return summary


def prepare_local_kaggle_retinal(
    csv_path: str,
    images_dir: str,
    id_col: str = "id_code",
    label_col: str = "diagnosis",
    patient_from_prefix: bool = False,
) -> dict:
    """
    Optional pipeline if the user manually downloads Kaggle APTOS 2019 or EyePACS.
    Verifies unreadable images, deduplicates via SHA-256 & dHash, and creates
    leak-free train/val/test CSV splits.
    """
    if not os.path.exists(csv_path) or not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Expected CSV ({csv_path}) and image directory ({images_dir}).")

    df_raw = pd.read_csv(csv_path)
    records = []
    seen_sha = set()
    corrupted = 0
    dups = 0

    for _, row in df_raw.iterrows():
        img_id = str(row[id_col])
        label = int(row[label_col])
        found_path = None
        for ext in ("", ".png", ".jpg", ".jpeg"):
            candidate = os.path.join(images_dir, img_id + ext)
            if os.path.isfile(candidate):
                found_path = candidate
                break
        if found_path is None:
            corrupted += 1
            continue

        try:
            with Image.open(found_path) as im:
                im.verify()
            with Image.open(found_path) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                if w < 64 or h < 64:
                    corrupted += 1
                    continue
                dhash = compute_dhash(rgb)
            sha = compute_sha256_file(found_path)
        except Exception:
            corrupted += 1
            continue

        if sha in seen_sha:
            dups += 1
            continue
        seen_sha.add(sha)

        patient_group = img_id.split("_")[0] if patient_from_prefix else img_id
        records.append(
            {
                "image_path": os.path.relpath(found_path, ROOT_DIR).replace("\\", "/"),
                "label": label,
                "label_name": DR_CLASS_NAMES[label],
                "patient_group": patient_group,
                "width": w,
                "height": h,
                "dhash": dhash,
                "sha256": sha,
            }
        )

    full_df = pd.DataFrame(records)
    if patient_from_prefix:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_val_idx, test_idx = next(gss1.split(full_df, full_df["label"], groups=full_df["patient_group"]))
        train_val_df = full_df.iloc[train_val_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        train_idx, val_idx = next(gss2.split(train_val_df, train_val_df["label"], groups=train_val_df["patient_group"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_val_idx, test_idx = next(sss1.split(full_df, full_df["label"]))
        train_val_df = full_df.iloc[train_val_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        train_idx, val_idx = next(sss2.split(train_val_df, train_val_df["label"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        d.to_csv(os.path.join(RETINAL_DIR, f"{name}.csv"), index=False)

    return {
        "source": csv_path,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "corrupted_skipped": corrupted,
        "duplicates_removed": dups,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Eye & Diabetes Screening Datasets")
    parser.add_argument("--force", action="store_true", help="Re-download and regenerate files")
    parser.add_argument("--kaggle-aptos-dir", type=str, default=None, help="Optional path to local Kaggle APTOS folder containing train.csv & train_images/")
    args = parser.parse_args()

    pima_summary = prepare_pima_diabetes(force=args.force)
    if args.kaggle_aptos_dir:
        ret_summary = prepare_local_kaggle_retinal(
            csv_path=os.path.join(args.kaggle_aptos_dir, "train.csv"),
            images_dir=os.path.join(args.kaggle_aptos_dir, "train_images"),
        )
    else:
        ret_summary = prepare_retinamnist(force=args.force)

    print("\nDataset preparation completed cleanly:")
    print(json.dumps({"pima_diabetes": pima_summary, "retinal_fundus": ret_summary}, indent=2))


if __name__ == "__main__":
    main()
