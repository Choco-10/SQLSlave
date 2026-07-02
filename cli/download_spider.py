"""Download the official Spider dataset archive and convert it.

Run with:

    python -m cli.download_spider

This script downloads the Spider dataset from the official Spider page
(Google Drive archive), extracts all files into the fixed `spider/`
directory, normalizes the key filenames needed by the converter, and
writes the converted JSONL files into the fixed `converted_spider/`
directory.
"""
from __future__ import annotations

import logging
import importlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

from core.dataset import convert_spider_dataset

try:
    gdown = importlib.import_module("gdown")
except Exception:  # pragma: no cover - optional if requirements are not installed yet
    gdown = None


logger = logging.getLogger(__name__)

SPIDER_ROOT = Path("spider")
OUTPUT_DIR = Path("converted_spider")
SPIDER_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download"
TRAIN_CANDIDATES = ("train_spider.json", "train.json")
DEV_CANDIDATES = ("dev.json", "validation.json", "dev_spider.json", "dev_spider_train.json")
TABLES_CANDIDATES = ("tables.json",)


def _download_google_drive_file(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if gdown is not None:
        downloaded = gdown.download(id=file_id, output=str(destination), quiet=False)
        if downloaded:
            return
        raise RuntimeError("gdown did not produce a downloadable Spider archive")

    session = requests.Session()
    response = session.get(
        GOOGLE_DRIVE_DOWNLOAD_URL,
        params={"id": file_id, "export": "download"},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(
            GOOGLE_DRIVE_DOWNLOAD_URL,
            params={"id": file_id, "export": "download", "confirm": token},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _extract_archive(archive_path: Path, spider_root: Path) -> None:
    spider_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        member_names = archive.namelist()
        logger.info("Archive contains %s entries", len(member_names))

        needed_members: dict[str, str] = {}
        for member_name in member_names:
            if member_name.endswith("/"):
                continue
            base_name = Path(member_name).name
            if base_name in TABLES_CANDIDATES and "tables.json" not in needed_members:
                needed_members["tables.json"] = member_name
            elif base_name in TRAIN_CANDIDATES and "train_spider.json" not in needed_members:
                needed_members["train_spider.json"] = member_name
            elif base_name in DEV_CANDIDATES and "dev.json" not in needed_members:
                needed_members["dev.json"] = member_name

        for target_name, member_name in needed_members.items():
            target_path = spider_root / target_name
            logger.info("Extracting %s -> %s", member_name, target_path)
            with archive.open(member_name, "r") as source, target_path.open("wb") as target:
                target.write(source.read())

        database_members = [
            m for m in member_names
            if "/database/" in m and m.endswith(".sqlite")
        ]
        if database_members:
            logger.info("Extracting %d SQLite databases …", len(database_members))
            for member_name in database_members:
                rel = member_name
                parts = Path(member_name).parts
                try:
                    db_idx = parts.index("database")
                    rel_path = Path(*parts[db_idx:])
                except ValueError:
                    rel_path = Path(member_name).name

                target_path = spider_root / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not target_path.exists():
                    with archive.open(member_name, "r") as source, target_path.open("wb") as target:
                        target.write(source.read())
            logger.info("Database extraction complete")
        else:
            logger.warning("No SQLite databases found in archive")

    missing = [name for name in ("tables.json", "train_spider.json", "dev.json") if not (spider_root / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required Spider files after extraction: {', '.join(missing)}")


def _find_file(spider_root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        direct = spider_root / candidate
        if direct.exists():
            return direct
        matches = sorted(spider_root.rglob(candidate))
        if matches:
            return matches[0]
    return None


def _copy_if_needed(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def download_spider_dataset(spider_root: Path, force: bool = False) -> None:
    spider_root.mkdir(parents=True, exist_ok=True)

    expected_files = [spider_root / "tables.json", spider_root / "train_spider.json", spider_root / "dev.json"]
    db_dir = spider_root / "database"
    databases_present = db_dir.exists() and any(db_dir.iterdir())

    if all(path.exists() for path in expected_files) and databases_present and not force:
        logger.info("Spider dataset already present under %s", spider_root)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "spider_dataset.zip"
        logger.info("Downloading official Spider dataset archive")
        _download_google_drive_file(SPIDER_FILE_ID, archive_path)
        logger.info("Extracting Spider dataset archive")
        _extract_archive(archive_path, spider_root)

    tables_source = _find_file(spider_root, TABLES_CANDIDATES)
    if tables_source is None:
        raise RuntimeError("Could not find tables.json in the extracted Spider dataset")
    _copy_if_needed(tables_source, spider_root / "tables.json")

    train_source = _find_file(spider_root, TRAIN_CANDIDATES)
    if train_source is None:
        raise RuntimeError("Could not find a Spider training split in the extracted dataset")
    _copy_if_needed(train_source, spider_root / "train_spider.json")

    dev_source = _find_file(spider_root, DEV_CANDIDATES)
    if dev_source is None:
        raise RuntimeError("Could not find a Spider development split in the extracted dataset")
    _copy_if_needed(dev_source, spider_root / "dev.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    download_spider_dataset(SPIDER_ROOT)
    train_path, validation_path = convert_spider_dataset(spider_root=SPIDER_ROOT, output_dir=OUTPUT_DIR)
    print(f"Wrote {train_path}")
    print(f"Wrote {validation_path}")


if __name__ == "__main__":
    main()