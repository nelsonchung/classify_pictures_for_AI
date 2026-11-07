"""Prepare COCO val2017 and play the images grouped by annotated category."""

import argparse
import fcntl
import json
import re
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent
INSTANCE_JSON = "annotations/instances_val2017.json"
ANNOTATION_FILES = [
    f"annotations/{kind}_{split}2017.json"
    for kind in ("instances", "captions", "person_keypoints")
    for split in ("train", "val")
]
JPEG = re.compile(r"[0-9]{12}\.jpg", re.ASCII)


def write_file(source, destination, expected_size):
    """Publish the file only after the download or ZIP member was read in full."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = NamedTemporaryFile(dir=destination.parent, delete=False)
    temporary = Path(target.name)
    try:
        with target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            if expected_size is not None and target.tell() != expected_size:
                raise OSError(f"檔案大小不符：{destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def download(root, remote_path):
    destination = root / remote_path.rsplit("/", 1)[-1]
    if destination.is_file():
        return destination

    for base in (
        "https://images.cocodataset.org",
        "https://s3.amazonaws.com/images.cocodataset.org",
    ):
        print(f"下載 {destination.name}（{base}）", flush=True)
        try:
            with urlopen(f"{base}/{remote_path}", timeout=30) as response:
                length = response.headers.get("Content-Length")
                write_file(response, destination, int(length) if length else None)
        except (URLError, TimeoutError, ConnectionError) as error:
            print(f"下載失敗：{error}", file=sys.stderr)
        else:
            return destination
    raise OSError(f"無法下載 {destination.name}")


def extract_missing(root, remote_path, names):
    missing = [
        name for name in names
        if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]
    if not missing:
        return

    archive = download(root, remote_path)
    print(f"解壓縮 {archive.name}（{len(missing)} 個檔案）", flush=True)
    try:
        with ZipFile(archive) as zipped:
            for name in missing:
                member = zipped.getinfo(name)
                with zipped.open(member) as source:
                    write_file(source, root / name, member.file_size)
    except (BadZipFile, KeyError) as error:
        raise ValueError(f"{archive.name} 損毀或缺少預期檔案：{error}") from error


def classify(data, filenames):
    image_ids_by_category = {category["id"]: set() for category in data["categories"]}
    labeled_images = set()
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        image_ids_by_category[annotation["category_id"]].add(image_id)
        labeled_images.add(image_id)

    categories = []
    for category in sorted(data["categories"], key=lambda row: row["id"]):
        images = sorted(filenames[image_id] for image_id in image_ids_by_category[category["id"]])
        categories.append({"name": category["name"], "images": images})

    return {
        "image_count": len(filenames),
        "unlabeled_count": len(filenames.keys() - labeled_images),
        "categories": categories,
    }


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".coco-prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        instance_file = root / INSTANCE_JSON
        annotation_zip = root / "annotations_trainval2017.zip"
        if annotation_zip.is_file() or not instance_file.is_file() or instance_file.stat().st_size == 0:
            extract_missing(root, "annotations/annotations_trainval2017.zip", ANNOTATION_FILES)
        with instance_file.open(encoding="utf-8") as source:
            data = json.load(source)
        filenames = {row["id"]: row["file_name"] for row in data["images"]}
        if not filenames or not all(JPEG.fullmatch(name) for name in filenames.values()):
            raise ValueError("COCO 標註檔的圖片清單無效")
        extract_missing(root, "zips/val2017.zip", [f"val2017/{name}" for name in filenames.values()])
        return classify(data, filenames), frozenset(filenames.values())


def serve(root, index, filenames, host, port):
    page = (ROOT / "player.html").read_bytes()
    catalog = json.dumps(index, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, content_type, length):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.reply("text/html; charset=utf-8", len(page))
                self.wfile.write(page)
            elif path == "/catalog.json":
                self.reply("application/json; charset=utf-8", len(catalog))
                self.wfile.write(catalog)
            elif path.startswith("/images/") and path[8:] in filenames:
                try:
                    source = (root / "val2017" / path[8:]).open("rb")
                except FileNotFoundError:
                    self.send_error(404)
                    return
                with source:
                    self.reply("image/jpeg", source.seek(0, 2))
                    source.seek(0)
                    shutil.copyfileobj(source, self.wfile)
            else:
                self.send_error(404)

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(f"播放器：http://{host}:{server.server_port}/", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT, help="ZIP 與解壓縮資料的位置")
    parser.add_argument("--list", action="store_true", help="列出類別與圖片數，不啟動播放器")
    parser.add_argument("--host", default="127.0.0.1", help="播放器的監聽位址")
    parser.add_argument("--port", default=8765, type=int, help="播放器的連接埠")
    args = parser.parse_args()
    root = args.data_dir.resolve()

    try:
        index, filenames = prepare(root)
        categories = [category for category in index["categories"] if category["images"]]
        print(f"{index['image_count']} 張圖片，{len(categories)} 個有圖片的類別，"
              f"{index['unlabeled_count']} 張沒有物件標註。", flush=True)
        if args.list:
            for category in index["categories"]:
                print(f"{category['name']:<16} {len(category['images']):>4}")
        else:
            serve(root, index, filenames, args.host, args.port)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"錯誤：{error}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
