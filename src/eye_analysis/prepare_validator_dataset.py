"""
Dataset Preparation & Leak-Free Splitting for the Human Retinal Fundus Image Validator.

Builds a diverse binary dataset (`data/retinal_validator/{train,val,test}.csv` plus
`external_messidor2_pos.csv` and `validator_dataset_summary.json`):
  - label = 1: Valid Human Retinal Fundus Image
      * Multi-center EyePACS (MIT), IDRiD (CC BY 4.0), DeepDRiD/RetinaMNIST (CC BY 4.0),
        APTOS 2019, plus held-out Messidor-2 for external zero-shot validation
      * Includes both FOV-cropped fundus scans and raw uncropped black-bordered fundus scans
  - label = 0: Invalid or Non-Retinal Image
      * Cat & dog portraits and close-up cat/dog eye crops (`cvdl/oxford-pets` MIT,
        `Francesco/animals-ij5d2` CC, Wikimedia Commons CC0/CC-BY/CC-BY-SA)
      * Human anterior eye / iris close-ups (`fatdove/Iris_Database` MIT, eye crops from
        `LakoreAI/human-nonhuman-face-classification` MIT, Wikimedia Commons)
      * Human & non-human faces (`LakoreAI/human-nonhuman-face-classification` MIT)
      * Hard-negative reddish/circular medical lesions (`danjacobellis/dermamnist_224`)
        and chest radiographs (`hf-vision/chest-xray-pneumonia` CC BY 4.0)
      * Natural objects, landscapes, scenes (`uoft-cs/cifar100`)
      * Simulated presentation attacks (phone/monitor screen pixel-grid & matte printed eye photos)

Strict leak prevention:
  - Group-level splitting by source image ID BEFORE extracting eye/face crops so a portrait in train
    never places its cropped eye in val or test.
  - Exact SHA-256 and perceptual 8x8 dHash deduplication across train, val, and test splits.
  - Messidor-2 held strictly outside train/val for zero-shot external evaluation.
"""

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image, UnidentifiedImageError

from src.eye_analysis.dataset_prep import DATA_DIR, EXT_CACHE_DIR, RETINAL_DIR, ROOT_DIR, compute_dhash, download_file

VALIDATOR_DATA_DIR = os.path.join(DATA_DIR, "retinal_validator")
VALIDATOR_CACHE_DIR = os.path.join(EXT_CACHE_DIR, "validator")
VALIDATOR_IMG_DIR = os.path.join(VALIDATOR_DATA_DIR, "images")

CAT_BREED_PREFIXES = {
    "abyssinian", "bengal", "birman", "bombay", "british", "egyptian",
    "maine", "persian", "ragdoll", "russian", "siamese", "sphynx",
}

DATASET_LICENSES = {
    "retinal_eyepacs": "MIT (Diabetic Retinopathy Detection)",
    "retinal_idrid": "CC BY 4.0 (IDRiD)",
    "retinal_retinamnist_deepdrid": "CC BY 4.0 (DeepDRiD / RetinaMNIST)",
    "retinal_aptos2019": "Public Kaggle/HF mirror (APTOS 2019)",
    "retinal_messidor2": "External zero-shot benchmark (Messidor-2)",
    "cvdl/oxford-pets": "MIT",
    "Francesco/animals-ij5d2": "Creative Commons (CC)",
    "LakoreAI/human-nonhuman-face-classification": "MIT",
    "fatdove/Iris_Database": "MIT",
    "hf-vision/chest-xray-pneumonia": "CC BY 4.0",
    "danjacobellis/dermamnist_224": "CC BY 4.0 (MedMNIST / HAM10000)",
    "uoft-cs/cifar100": "Public academic benchmark (CIFAR-100)",
    "wikimedia_commons": "CC0 / Public Domain / CC BY / CC BY-SA",
}

NEGATIVE_PARQUET_SOURCES = {
    "pets_test": "https://huggingface.co/datasets/cvdl/oxford-pets/resolve/main/data/test-00000-of-00001.parquet",
    "animals_train": "https://huggingface.co/datasets/Francesco/animals-ij5d2/resolve/main/data/train-00000-of-00001-83ef17440fd25f6f.parquet",
    "animals_val": "https://huggingface.co/datasets/Francesco/animals-ij5d2/resolve/main/data/validation-00000-of-00001-876de533d76d48c6.parquet",
    "animals_test": "https://huggingface.co/datasets/Francesco/animals-ij5d2/resolve/main/data/test-00000-of-00001-77e7809ef7ac5cc0.parquet",
    "faces_train": "https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification/resolve/main/data/train-00000-of-00001.parquet",
    "faces_val": "https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification/resolve/main/data/validation-00000-of-00001.parquet",
    "faces_test": "https://huggingface.co/datasets/LakoreAI/human-nonhuman-face-classification/resolve/main/data/test-00000-of-00001.parquet",
    "derma_val": "https://huggingface.co/datasets/danjacobellis/dermamnist_224/resolve/main/data/validation-00000-of-00001.parquet",
    "derma_test": "https://huggingface.co/datasets/danjacobellis/dermamnist_224/resolve/main/data/test-00000-of-00001.parquet",
    "xray_test": "https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/resolve/main/data/test-00000-of-00001.parquet",
    "cifar100_test": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
}


def _split_by_key(key: str) -> str:
    """Deterministically hash a source group ID into train (68%), val (16%), test (16%)."""
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 68:
        return "train"
    if h < 84:
        return "val"
    return "test"


def _save_rgb_aspect_preserved(rgb_np: np.ndarray, out_path: str, max_side: int = 288) -> Optional[Dict]:
    """Save RGB image resized smoothly without stretching aspect ratio and return SHA-256 + dHash."""
    if rgb_np is None or rgb_np.ndim < 2 or min(rgb_np.shape[:2]) < 32:
        return None
    if rgb_np.ndim == 2:
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2RGB)
    elif rgb_np.shape[2] == 4:
        rgb_np = rgb_np[:, :, :3]

    h, w = rgb_np.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        rgb_np = cv2.resize(rgb_np, (max(32, int(round(w * scale))), max(32, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    pil_img = Image.fromarray(rgb_np.astype(np.uint8))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil_img.save(out_path, format="JPEG", quality=92)
    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "width": int(pil_img.size[0]),
        "height": int(pil_img.size[1]),
        "dhash": compute_dhash(pil_img),
        "sha256": sha,
    }


def _row_img_cell(row):
    """Extract raw image column value robustly across different Parquet schemas."""
    for col in ("image", "img", "image_bytes", "bytes"):
        if col in row and row[col] is not None:
            return row[col]
    return None


def _decode_parquet_image(img_cell) -> Optional[np.ndarray]:
    """Extract RGB numpy array cleanly from a Parquet image struct, bytes, or numpy array."""
    try:
        if isinstance(img_cell, dict) and img_cell.get("bytes") is not None:
            with Image.open(io.BytesIO(img_cell["bytes"])) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        if isinstance(img_cell, (bytes, bytearray)):
            with Image.open(io.BytesIO(img_cell)) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        if isinstance(img_cell, np.ndarray):
            arr = img_cell
            if arr.ndim == 2:
                return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[:, :, :3].astype(np.uint8)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    return None


def _build_record(
    rel_or_abs_path: str,
    label: int,
    split: str,
    subcategory: str,
    source: str,
    group_id: str,
    meta: Dict,
) -> Dict:
    rel = os.path.relpath(rel_or_abs_path, ROOT_DIR).replace("\\", "/") if os.path.isabs(rel_or_abs_path) else rel_or_abs_path.replace("\\", "/")
    return {
        "image_path": rel,
        "label": int(label),
        "label_name": "valid_human_retinal_fundus" if int(label) == 1 else "invalid_non_retinal",
        "split": split,
        "subcategory": subcategory,
        "dataset_source": source,
        "group_id": group_id,
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "dhash": meta["dhash"],
        "sha256": meta["sha256"],
    }


def _add_dark_vignette_variant(rgb_np: np.ndarray) -> np.ndarray:
    """Apply realistic dark-room flash vignette around an animal/anterior eye close-up (hard negative)."""
    h, w = rgb_np.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    r = np.hypot(xx - cx, yy - cy) / max(1.0, 0.44 * min(h, w))
    mask = np.clip(1.0 - np.clip((r - 0.65) / 0.35, 0.0, 1.0), 0.04, 1.0)
    out = np.clip(rgb_np.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)
    return out


def _add_screen_attack_variant(rgb_np: np.ndarray, is_print: bool = False, seed: int = 0) -> np.ndarray:
    """Render realistic display-replay (Moire + bezel) or matte printed paper attack onto a non-retinal eye image."""
    h, w = rgb_np.shape[:2]
    rng = np.random.default_rng(seed)
    out = rgb_np.astype(np.int16)
    if is_print:
        noise = rng.normal(0, 7, out.shape).astype(np.int16)
        out = np.clip(out + noise, 0, 255).astype(np.uint8)
        out = cv2.GaussianBlur(out, (3, 3), 0)
        # Mild paper border frame
        out = cv2.copyMakeBorder(out, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=(232, 228, 220))
        return out

    # Digital display RGB stripe/Moire + device bezel
    y_coords = np.arange(h)[:, None]
    x_coords = np.arange(w)[None, :]
    moire = (12 * np.sin(2.0 * np.pi * y_coords / 4.0) + 7 * np.sin(2.0 * np.pi * x_coords / 3.0)).astype(np.int16)
    out = np.clip(out + moire[:, :, None], 0, 255).astype(np.uint8)
    # Cool LED cast + phone bezel frame
    out[:, :, 2] = np.clip(out[:, :, 2].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    bx1, by1 = max(4, int(w * 0.08)), max(4, int(h * 0.06))
    bx2, by2 = w - bx1, h - by1
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (18, 18, 22), max(4, min(h, w) // 28))
    return out


def _collect_positive_fundus(download_raw: bool = True) -> Dict[str, List[Dict]]:
    """Collect diverse Valid Human Retinal Fundus samples across EyePACS, IDRiD, DeepDRiD, APTOS, plus raw uncropped scans."""
    out: Dict[str, List[Dict]] = {"train": [], "val": [], "test": [], "external_messidor2": []}

    # 1. Pre-split processed fundus scans (guarantees exact dataset split isolation)
    caps = {"train": 1500, "val": 520, "test": 620}
    for sp, cap in caps.items():
        csv_p = os.path.join(RETINAL_DIR, f"{sp}.csv")
        if not os.path.exists(csv_p):
            continue
        df = pd.read_csv(csv_p)
        # Stratify across grades 0..4 so all DR severities are represented equally well
        per_grade = max(20, cap // 5)
        sampled_parts = []
        for g in range(5):
            sub = df[df["label"] == g]
            if len(sub) > per_grade:
                sub = sub.sample(n=per_grade, random_state=42 + g)
            sampled_parts.append(sub)
        df_sub = pd.concat(sampled_parts, ignore_index=True)
        for row in df_sub.to_dict("records"):
            abs_p = os.path.join(ROOT_DIR, str(row["image_path"]))
            if not os.path.exists(abs_p):
                continue
            out[sp].append({
                "image_path": str(row["image_path"]).replace("\\", "/"),
                "label": 1,
                "label_name": "valid_human_retinal_fundus",
                "split": sp,
                "subcategory": f"retinal_grade{int(row['label'])}",
                "dataset_source": str(row.get("dataset_source", "retinal_multicenter")),
                "group_id": str(row.get("patient_group", row["image_path"])),
                "width": int(row.get("width", 224)),
                "height": int(row.get("height", 224)),
                "dhash": str(row.get("dhash", "")),
                "sha256": str(row.get("sha256", "")),
            })

    # 2. Raw uncropped black-bordered clinical fundus images from cached parquets (strict split mapping!)
    raw_specs = [
        ("train", "eyepacs_train0.parquet", "eyepacs", 200),
        ("train", "eyepacs_train2.parquet", "eyepacs", 200),
        ("train", "idrid_train.parquet", "idrid", 150),
        ("val", "eyepacs_val.parquet", "eyepacs", 150),
        ("test", "eyepacs_test.parquet", "eyepacs", 180),
        ("test", "idrid_test.parquet", "idrid", 80),
    ]
    for sp, pq_name, src, take_n in raw_specs:
        pq_path = os.path.join(EXT_CACHE_DIR, pq_name)
        if not os.path.exists(pq_path):
            continue
        tbl = pq.read_table(pq_path)
        df_raw = tbl.to_pandas().sample(n=min(take_n, tbl.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_raw.iterrows():
            rgb = _decode_parquet_image(row["image"])
            if rgb is None:
                continue
            out_path = os.path.join(VALIDATOR_IMG_DIR, sp, "pos_raw", f"raw_{src}_{os.path.splitext(pq_name)[0]}_{i:04d}.jpg")
            if not os.path.exists(out_path):
                meta = _save_rgb_aspect_preserved(rgb, out_path, max_side=320)
            else:
                with Image.open(out_path) as im:
                    with open(out_path, "rb") as f:
                        sha = hashlib.sha256(f.read()).hexdigest()
                    meta = {"width": im.size[0], "height": im.size[1], "dhash": compute_dhash(im), "sha256": sha}
            if meta is None:
                continue
            out[sp].append(_build_record(out_path, 1, sp, "retinal_raw_uncropped", f"{src}_raw", f"raw_{pq_name}_{i}", meta))

    # 3. External Messidor-2 strictly held out
    mess_csv = os.path.join(RETINAL_DIR, "external_messidor2.csv")
    if os.path.exists(mess_csv):
        df_m = pd.read_csv(mess_csv)
        for row in df_m.to_dict("records"):
            out["external_messidor2"].append({
                "image_path": str(row["image_path"]).replace("\\", "/"),
                "label": 1,
                "label_name": "valid_human_retinal_fundus",
                "split": "external_messidor2",
                "subcategory": f"retinal_messidor2_g{int(row['label'])}",
                "dataset_source": "messidor2",
                "group_id": str(row.get("patient_group", row["image_path"])),
                "width": int(row.get("width", 320)),
                "height": int(row.get("height", 320)),
                "dhash": str(row.get("dhash", "")),
                "sha256": str(row.get("sha256", "")),
            })

    return out


def _harvest_wikimedia_eyes(neg_records: Dict[str, List[Dict]], skip_download: bool = False) -> None:
    """Fetch open-licensed (CC0/PD/CC-BY/CC-BY-SA) macro cat, dog, animal, and human iris closeups."""
    manifest_cache = os.path.join(VALIDATOR_CACHE_DIR, "wikimedia_manifest.json")
    items = []
    if os.path.exists(manifest_cache):
        with open(manifest_cache, "r", encoding="utf-8") as f:
            items = json.load(f)
    elif not skip_download:
        queries = [
            ("cat eye", "cat_eye"),
            ("kitten eye closeup", "cat_eye"),
            ("dog eye closeup", "dog_eye"),
            ("puppy eye", "dog_eye"),
            ("owl lizard animal eye", "animal_eye"),
            ("human iris eye macro", "human_iris"),
        ]
        seen_titles = set()
        for q, subcat in queries:
            try:
                resp = requests.get(
                    "https://commons.wikimedia.org/w/api.php",
                    params={
                        "action": "query",
                        "generator": "search",
                        "gsrsearch": f"filetype:bitmap {q}",
                        "gsrnamespace": 6,
                        "gsrlimit": 24,
                        "prop": "imageinfo",
                        "iiprop": "url|mime|extmetadata",
                        "iiurlwidth": 448,
                        "format": "json",
                    },
                    headers={"User-Agent": "AI-Eye-Analysis-Validator/1.0"},
                    timeout=15,
                )
                pages = ((resp.json().get("query") or {}).get("pages") or {}) if resp.status_code == 200 else {}
                for p in pages.values():
                    title = p.get("title", "")
                    if not title or title in seen_titles:
                        continue
                    ii = (p.get("imageinfo") or [{}])[0]
                    url = ii.get("thumburl") or ii.get("url")
                    mime = ii.get("mime", "")
                    lic = ((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value", "")
                    if not url or "svg" in mime or not any(k in lic.lower() for k in ("cc0", "cc by", "cc-by", "public domain")):
                        continue
                    seen_titles.add(title)
                    items.append({"title": title, "url": url, "subcat": subcat, "license": lic})
            except Exception:
                pass
        os.makedirs(VALIDATOR_CACHE_DIR, exist_ok=True)
        with open(manifest_cache, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

    for idx, it in enumerate(items):
        gid = f"wiki_{hashlib.md5(it['title'].encode('utf-8')).hexdigest()[:10]}"
        sp = _split_by_key(gid)
        out_path = os.path.join(VALIDATOR_IMG_DIR, sp, "neg_wiki", f"{gid}_{it['subcat']}.jpg")
        if not os.path.exists(out_path):
            if skip_download:
                continue
            try:
                r = requests.get(it["url"], headers={"User-Agent": "AI-Eye-Analysis-Validator/1.0"}, timeout=15)
                if r.status_code != 200:
                    continue
                with Image.open(io.BytesIO(r.content)) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                meta = _save_rgb_aspect_preserved(rgb, out_path, max_side=288)
            except Exception:
                meta = None
        else:
            with Image.open(out_path) as im:
                with open(out_path, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
                meta = {"width": im.size[0], "height": im.size[1], "dhash": compute_dhash(im), "sha256": sha}
        if meta:
            neg_records[sp].append(_build_record(out_path, 0, sp, it["subcat"], "wikimedia_commons", gid, meta))


def _save_and_record(
    rgb_np: np.ndarray,
    out_path: str,
    split: str,
    subcategory: str,
    source: str,
    group_id: str,
    neg_records: Dict[str, List[Dict]],
) -> bool:
    if os.path.exists(out_path):
        try:
            with Image.open(out_path) as im:
                with open(out_path, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()
                meta = {"width": im.size[0], "height": im.size[1], "dhash": compute_dhash(im), "sha256": sha}
        except Exception:
            meta = _save_rgb_aspect_preserved(rgb_np, out_path)
    else:
        meta = _save_rgb_aspect_preserved(rgb_np, out_path)
    if meta is None:
        return False
    neg_records[split].append(_build_record(out_path, 0, split, subcategory, source, group_id, meta))
    return True


def _collect_negative_sources(download_missing: bool = True) -> Dict[str, List[Dict]]:
    """Download & ingest diverse non-retinal negatives across all required rejection categories."""
    neg_records: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    os.makedirs(VALIDATOR_CACHE_DIR, exist_ok=True)

    # Load OpenCV cascades once for extracting close-up cat eyes, dog eyes, and human eyes
    cat_cascade_p = os.path.join(cv2.data.haarcascades, "haarcascade_frontalcatface_extended.xml")
    eye_cascade_p = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    cat_cascade = cv2.CascadeClassifier(cat_cascade_p) if os.path.exists(cat_cascade_p) else None
    eye_cascade = cv2.CascadeClassifier(eye_cascade_p) if os.path.exists(eye_cascade_p) else None

    def _get_pq(name: str) -> Optional[str]:
        p = os.path.join(VALIDATOR_CACHE_DIR, f"{name}.parquet")
        if not os.path.exists(p):
            if not download_missing:
                return None
            download_file(NEGATIVE_PARQUET_SOURCES[name], p, timeout=360)
        return p

    # 1. Oxford-IIIT Pets (MIT, cat/dog breeds) -> portraits + bbox/cascade cat/dog eye crops + dark vignette animal eyes
    pets_p = _get_pq("pets_test")
    if pets_p:
        df_pets = pq.read_table(pets_p).to_pandas().reset_index(drop=True)
        # In cvdl/oxford-pets, ~12/37 breeds are cats; detect which class ID (0 or 1) has smaller count (~32%) = cats
        cat_cls_id = None
        if "class" in df_pets.columns:
            vc = df_pets["class"].value_counts()
            if len(vc) >= 2:
                cat_cls_id = int(vc.index[-1])
        df_pets = df_pets.sample(n=min(620, len(df_pets)), random_state=42).reset_index(drop=True)
        for i, row in df_pets.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            gid = f"pet_{i}"
            sp = _split_by_key(gid)
            lbl_str = str(row.get("label_name", row.get("label", ""))).lower()
            is_cat = (
                (cat_cls_id is not None and "class" in row and int(row["class"]) == cat_cls_id)
                or any(c in lbl_str for c in CAT_BREED_PREFIXES)
                or (cat_cascade is not None and len(cat_cascade.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 3)) > 0)
            )
            full_sub = "cat_portrait" if is_cat else "dog_portrait"
            eye_sub = "cat_eye" if is_cat else "dog_eye"

            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_pets", f"{gid}_{full_sub}.jpg"),
                sp,
                full_sub,
                "cvdl/oxford-pets",
                gid,
                neg_records,
            )

            # Extract realistic close-up animal eye crop (head bbox upper eye region -> cat cascade -> Haar eye)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            crop_rgb = None
            bbox = row.get("bbox") if "bbox" in row else None
            if bbox is not None and hasattr(bbox, "__len__") and len(bbox) == 4:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                bw, bh = bx2 - bx1, by2 - by1
                if bw >= 40 and bh >= 40:
                    # Focus closely around the eye level (single eye left/right or binocular bridge)
                    ey1 = max(0, by1 + int(bh * 0.14))
                    ey2 = min(h, by1 + int(bh * 0.56))
                    ex1 = max(0, bx1 + (int(bw * 0.08) if i % 2 == 0 else int(bw * 0.44)))
                    ex2 = min(w, bx1 + (int(bw * 0.56) if i % 2 == 0 else int(bw * 0.92)))
                    if ex2 - ex1 >= 34 and ey2 - ey1 >= 34:
                        crop_rgb = rgb[ey1:ey2, ex1:ex2]
            if crop_rgb is None and is_cat and cat_cascade is not None:
                faces = cat_cascade.detectMultiScale(gray, 1.08, 3, minSize=(40, 40))
                if len(faces) > 0:
                    fx, fy, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
                    ey1 = max(0, fy + int(fh * 0.14))
                    ey2 = min(h, fy + int(fh * 0.60))
                    ex1 = max(0, fx + (int(fw * 0.08) if i % 2 == 0 else int(fw * 0.45)))
                    ex2 = min(w, fx + (int(fw * 0.55) if i % 2 == 0 else int(fw * 0.92)))
                    if ex2 - ex1 >= 36 and ey2 - ey1 >= 36:
                        crop_rgb = rgb[ey1:ey2, ex1:ex2]
            if crop_rgb is None and eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(cv2.equalizeHist(gray), 1.08, 3, minSize=(24, 24))
                if len(eyes) > 0:
                    ex, ey, ew, eh = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[0]
                    pad = int(max(ew, eh) * 0.55)
                    crop_rgb = rgb[max(0, ey - pad) : min(h, ey + eh + pad), max(0, ex - pad) : min(w, ex + ew + pad)]
            if crop_rgb is None and min(h, w) >= 120:
                crop_rgb = rgb[int(h * 0.15) : int(h * 0.58), int(w * 0.22) : int(w * 0.78)]

            if crop_rgb is not None and min(crop_rgb.shape[:2]) >= 34:
                _save_and_record(
                    crop_rgb,
                    os.path.join(VALIDATOR_IMG_DIR, sp, "neg_animal_eyes", f"{gid}_{eye_sub}.jpg"),
                    sp,
                    eye_sub,
                    "cvdl/oxford-pets",
                    gid,
                    neg_records,
                )
                if i % 3 == 0:
                    vig = _add_dark_vignette_variant(crop_rgb)
                    _save_and_record(
                        vig,
                        os.path.join(VALIDATOR_IMG_DIR, sp, "neg_animal_eyes", f"{gid}_{eye_sub}_vig.jpg"),
                        sp,
                        eye_sub,
                        "cvdl/oxford-pets",
                        gid,
                        neg_records,
                    )

    # 2. Francesco/animals-ij5d2 (CC license, already split train/val/test)
    for sp, pq_key, cap in [("train", "animals_train", 260), ("val", "animals_val", 80), ("test", "animals_test", 100)]:
        ap = _get_pq(pq_key)
        if not ap:
            continue
        tbl_a = pq.read_table(ap)
        df_a = tbl_a.to_pandas().sample(n=min(cap, tbl_a.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_a.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            gid = f"animal_{sp}_{i}"
            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_animals", f"{gid}.jpg"),
                sp,
                "animal_other",
                "Francesco/animals-ij5d2",
                gid,
                neg_records,
            )

    # 3. LakoreAI/human-nonhuman-face-classification (MIT, pre-split train/val/test)
    #    -> yields Human Face portraits AND extracted Human Iris / Anterior Eye crops + screen/print attack variants
    for sp, pq_key, cap in [("train", "faces_train", 460), ("val", "faces_val", 150), ("test", "faces_test", 180)]:
        fp = _get_pq(pq_key)
        if not fp:
            continue
        tbl_f = pq.read_table(fp)
        df_f = tbl_f.to_pandas().sample(n=min(cap, tbl_f.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_f.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            gid = f"face_{sp}_{i}"
            is_human = int(row.get("label", 0)) == 1
            sub = "human_face" if is_human else "nonhuman_face"
            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_faces", f"{gid}_{sub}.jpg"),
                sp,
                sub,
                "LakoreAI/human-nonhuman-face-classification",
                gid,
                neg_records,
            )

            if eye_cascade is not None:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                eyes = eye_cascade.detectMultiScale(cv2.equalizeHist(gray), 1.08, 3, minSize=(20, 20))
                if len(eyes) > 0:
                    ex, ey, ew, eh = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[0]
                    h, w = gray.shape
                    pad = int(max(ew, eh) * 0.45)
                    eye_crop = rgb[max(0, ey - pad) : min(h, ey + eh + pad), max(0, ex - pad) : min(w, ex + ew + pad)]
                    if min(eye_crop.shape[:2]) >= 34:
                        esub = "human_iris" if is_human else "nonhuman_eye"
                        _save_and_record(
                            eye_crop,
                            os.path.join(VALIDATOR_IMG_DIR, sp, "neg_human_eyes", f"{gid}_{esub}.jpg"),
                            sp,
                            esub,
                            "LakoreAI/human-nonhuman-face-classification",
                            gid,
                            neg_records,
                        )
                        if is_human and i % 4 == 0:
                            # Screen replay / matte print simulation on human anterior eye (same group split)
                            is_prt = (i % 8 == 0)
                            atk = _add_screen_attack_variant(eye_crop, is_print=is_prt, seed=i)
                            _save_and_record(
                                atk,
                                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_screen_print", f"{gid}_{'print' if is_prt else 'screen'}.jpg"),
                                sp,
                                "printed_eye" if is_prt else "screen_eye",
                                "LakoreAI_attack_sim",
                                gid,
                                neg_records,
                            )

    # 4. fatdove/Iris_Database (MIT) — true close-up macro human anterior iris images (45 samples split cleanly)
    iris_cache = os.path.join(VALIDATOR_CACHE_DIR, "iris_db_list.json")
    iris_list = []
    if os.path.exists(iris_cache):
        with open(iris_cache, "r", encoding="utf-8") as f:
            iris_list = json.load(f)
    elif download_missing:
        try:
            r_meta = requests.get("https://huggingface.co/api/datasets/fatdove/Iris_Database", timeout=15)
            if r_meta.status_code == 200:
                iris_list = [
                    s["rfilename"]
                    for s in r_meta.json().get("siblings", [])
                    if s["rfilename"].endswith((".png", ".jpg"))
                ][:24]
                with open(iris_cache, "w", encoding="utf-8") as f:
                    json.dump(iris_list, f)
        except Exception:
            pass
    for idx, fname in enumerate(iris_list):
        gid = f"irisdb_{fname}"
        sp = _split_by_key(gid)
        out_path = os.path.join(VALIDATOR_IMG_DIR, sp, "neg_irisdb", f"{idx:03d}.jpg")
        if not os.path.exists(out_path):
            if not download_missing:
                continue
            try:
                u = f"https://huggingface.co/datasets/fatdove/Iris_Database/resolve/main/{fname}"
                resp = requests.get(u, timeout=25)
                if resp.status_code != 200:
                    continue
                with Image.open(io.BytesIO(resp.content)) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                _save_and_record(rgb, out_path, sp, "human_iris", "fatdove/Iris_Database", gid, neg_records)
            except Exception:
                pass
        else:
            _save_and_record(np.zeros((1, 1, 3), np.uint8), out_path, sp, "human_iris", "fatdove/Iris_Database", gid, neg_records)

    # 5. Open-licensed Wikimedia Commons Cat, Dog, Animal & Human Iris closeups
    _harvest_wikimedia_eyes(neg_records, skip_download=not download_missing)

    # 6. Medical Non-Retinal Hard Negatives: DermaMNIST (warm red/amber circular lesions) & Chest X-Ray
    #    Partition DermaMNIST strictly: val.parquet -> train + val (group-hashed), test.parquet -> test only!
    d_val_p = _get_pq("derma_val")
    if d_val_p:
        tbl_dv = pq.read_table(d_val_p)
        df_dv = tbl_dv.to_pandas().sample(n=min(420, tbl_dv.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_dv.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            gid = f"derma_v_{i}"
            sp = "train" if i < 320 else "val"
            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_derma", f"{gid}.jpg"),
                sp,
                "medical_non_retinal_derma",
                "danjacobellis/dermamnist_224",
                gid,
                neg_records,
            )

    d_test_p = _get_pq("derma_test")
    if d_test_p:
        tbl_dt = pq.read_table(d_test_p)
        df_dt = tbl_dt.to_pandas().sample(n=min(150, tbl_dt.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_dt.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            gid = f"derma_t_{i}"
            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, "test", "neg_derma", f"{gid}.jpg"),
                "test",
                "medical_non_retinal_derma",
                "danjacobellis/dermamnist_224",
                gid,
                neg_records,
            )

    xray_p = _get_pq("xray_test")
    if xray_p:
        tbl_x = pq.read_table(xray_p)
        df_x = tbl_x.to_pandas().sample(n=min(300, tbl_x.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_x.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            gid = f"xray_{i}"
            sp = _split_by_key(gid)
            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_xray", f"{gid}.jpg"),
                sp,
                "medical_non_retinal_xray",
                "hf-vision/chest-xray-pneumonia",
                gid,
                neg_records,
            )

    # 7. General Natural Objects & Landscapes (`uoft-cs/cifar100`)
    cif_p = _get_pq("cifar100_test")
    if cif_p:
        tbl_c = pq.read_table(cif_p)
        df_c = tbl_c.to_pandas().sample(n=min(550, tbl_c.num_rows), random_state=42).reset_index(drop=True)
        for i, row in df_c.iterrows():
            rgb = _decode_parquet_image(_row_img_cell(row))
            if rgb is None:
                continue
            # Upscale tiny 32x32 natural patches gently with cubic interpolation so >= 64x64 minimum passes cleanly
            rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_CUBIC)
            gid = f"cifar100_{i}"
            sp = _split_by_key(gid)
            _save_and_record(
                rgb,
                os.path.join(VALIDATOR_IMG_DIR, sp, "neg_natural", f"{gid}.jpg"),
                sp,
                "natural_object_landscape",
                "uoft-cs/cifar100",
                gid,
                neg_records,
            )

    return neg_records


def prepare_validator_dataset(download_missing: bool = True) -> Dict:
    """Assemble clean, deduplicated, split-isolated train/val/test manifests for Retinal Validator."""
    os.makedirs(VALIDATOR_DATA_DIR, exist_ok=True)
    os.makedirs(VALIDATOR_IMG_DIR, exist_ok=True)

    print("[Validator Prep] Collecting positive human retinal fundus splits...", flush=True)
    pos_by_split = _collect_positive_fundus(download_raw=download_missing)

    print("[Validator Prep] Downloading & extracting negative non-retinal splits...", flush=True)
    neg_by_split = _collect_negative_sources(download_missing=download_missing)

    splits = {}
    seen_sha = set()
    seen_dhash_train = set()
    seen_dhash_val = set()
    leak_removed = 0

    for sp in ("train", "val", "test"):
        rows = pos_by_split[sp] + neg_by_split[sp]
        df_sp = pd.DataFrame(rows)
        kept = []
        for r in df_sp.to_dict("records"):
            sha = str(r.get("sha256", ""))
            dh = str(r.get("dhash", ""))
            if sha and sha in seen_sha:
                leak_removed += 1
                continue
            if sp == "val" and dh and dh in seen_dhash_train:
                leak_removed += 1
                continue
            if sp == "test" and dh and (dh in seen_dhash_train or dh in seen_dhash_val):
                leak_removed += 1
                continue
            if sha:
                seen_sha.add(sha)
            if sp == "train" and dh:
                seen_dhash_train.add(dh)
            elif sp == "val" and dh:
                seen_dhash_val.add(dh)
            kept.append(r)

        df_clean = pd.DataFrame(kept).sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_csv = os.path.join(VALIDATOR_DATA_DIR, f"{sp}.csv")
        df_clean.to_csv(out_csv, index=False)
        splits[sp] = df_clean

    # External Messidor-2 positive manifest for zero-shot recall benchmark
    df_ext = pd.DataFrame(pos_by_split.get("external_messidor2", []))
    if len(df_ext) > 0:
        df_ext.to_csv(os.path.join(VALIDATOR_DATA_DIR, "external_messidor2_pos.csv"), index=False)

    summary = {
        "dataset": "Human Retinal Fundus vs. Non-Retinal / Animal Eye Validation Benchmark",
        "licenses": DATASET_LICENSES,
        "splits": {sp: int(len(df)) for sp, df in splits.items()},
        "external_messidor2_pos": int(len(df_ext)),
        "cross_split_duplicates_removed": int(leak_removed),
        "class_balance": {
            sp: {
                "valid_human_retinal_fundus": int((df["label"] == 1).sum()),
                "invalid_non_retinal": int((df["label"] == 0).sum()),
            }
            for sp, df in splits.items()
        },
        "subcategory_breakdown": {
            sp: {str(k): int(v) for k, v in Counter(df["subcategory"]).items()}
            for sp, df in splits.items()
        },
    }
    with open(os.path.join(VALIDATOR_DATA_DIR, "validator_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Validator Prep] Completed cleanly:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare Multi-Source Retinal Fundus Validator Dataset")
    parser.add_argument("--no-download", action="store_true", help="Use cached files only without network downloads")
    args = parser.parse_args()
    prepare_validator_dataset(download_missing=not args.no_download)


if __name__ == "__main__":
    main()
