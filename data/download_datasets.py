"""
download_coco.py
────────────────
Resume-capable COCO val2017 downloader.
Agar beech mein ruk jaye toh dobara run karo — wahan se shuru karega jahan ruka tha.

Run:
    python data/download_coco.py
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "data" / "raw" / "coco"
COCO_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    {
        "url":      "http://images.cocodataset.org/zips/val2017.zip",
        "zip":      COCO_DIR / "val2017.zip",
        "out_dir":  COCO_DIR,
        "check":    COCO_DIR / "val2017",   # folder that should exist after extract
        "name":     "COCO val2017 images (~778 MB)",
    },
    {
        "url":      "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "zip":      COCO_DIR / "annotations_trainval2017.zip",
        "out_dir":  COCO_DIR,
        "check":    COCO_DIR / "annotations",
        "name":     "COCO annotations (~241 MB)",
    },
]


def download_resume(url: str, dest: Path, desc: str) -> bool:
    """
    Download with resume support.
    Uses HTTP Range header to continue partial downloads.
    Returns True if file is complete.
    """
    headers = {}
    existing_size = dest.stat().st_size if dest.exists() else 0

    # Get total size first
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            total_size = int(r.headers.get("Content-Length", 0))
    except Exception:
        total_size = 0

    # Already fully downloaded?
    if existing_size > 0 and total_size > 0 and existing_size >= total_size * 0.99:
        print(f"  [skip] {dest.name} already fully downloaded.")
        return True

    if existing_size > 0:
        print(f"  Resuming {dest.name} from {existing_size/1e6:.1f} MB...")
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
    else:
        print(f"  Downloading {desc}...")
        mode = "wb"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            remaining = total_size - existing_size if total_size else None
            with tqdm(total=remaining, initial=0,
                      unit="B", unit_scale=True,
                      desc=dest.name, miniters=1) as bar:
                with open(dest, mode) as f:
                    while True:
                        chunk = response.read(1024 * 256)  # 256 KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"  Download complete: {dest.stat().st_size/1e6:.1f} MB")
        return True

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Run this script again to resume.")
        return False


def extract_zip(zip_path: Path, out_dir: Path, check_dir: Path) -> bool:
    """Extract zip, skip if already extracted."""
    if check_dir.exists() and any(check_dir.iterdir()):
        print(f"  [skip] Already extracted: {check_dir.name}/")
        return True

    print(f"  Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            for m in tqdm(members, desc="Extracting", unit="file"):
                zf.extract(m, out_dir)
        print(f"  Extraction done.")
        return True
    except zipfile.BadZipFile:
        print(f"  [ERROR] Zip file is corrupt — deleting and re-downloading.")
        zip_path.unlink()
        return False


def main():
    print("=" * 56)
    print("  COCO val2017 Resume Downloader")
    print("=" * 56)

    for item in DOWNLOADS:
        print(f"\n── {item['name']} ──")

        # Download (with resume)
        ok = download_resume(item["url"], item["zip"], item["name"])
        if not ok:
            print("  Stopping. Run again to resume.")
            return

        # Extract
        ok = extract_zip(item["zip"], item["out_dir"], item["check"])
        if not ok:
            # Zip was corrupt, re-download
            ok = download_resume(item["url"], item["zip"], item["name"])
            if ok:
                extract_zip(item["zip"], item["out_dir"], item["check"])

    print("\n" + "="*56)
    print("  COCO download complete!")
    print("  Next step:")
    print("    python data/prepare_dataset.py")
    print("="*56)


if __name__ == "__main__":
    main()
