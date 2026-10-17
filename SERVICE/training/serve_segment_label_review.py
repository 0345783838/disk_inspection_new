"""Prepare a localhost review UI; human decisions never modify training masks."""
import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from curate_segment_masks import band_image, complete_component, ridge_image


def prepare(point_dataset, curated, queue, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    points = json.loads((point_dataset / "manifest.json").read_text(encoding="utf-8"))["records"]
    point_lookup = {r["source"]: r for r in points if r["origin"] == "factory"}
    masks = json.loads((curated / "manifest.json").read_text(encoding="utf-8"))
    mask_lookup = {(r["source"], r["band"]): r for r in masks}
    queue_bytes = queue.read_bytes()
    entries = json.loads(queue_bytes)
    items = []
    band_cache = {}
    for index, edge in enumerate(entries, 1):
        key = (edge["source"], edge["band"])
        if key not in band_cache:
            band_cache[key] = band_image(point_lookup[edge["source"]], edge["band"])
        band = band_cache[key]
        record = mask_lookup[key]
        proposal = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
        count, components, stats, centers = cv2.connectedComponentsWithStats(
            (proposal > 0).astype(np.uint8), 8)
        candidates = list(range(1, count))
        selected = min(candidates, key=lambda i: (
            abs(float(centers[i, 0]) - edge["x"]) +
            abs(int(stats[i, cv2.CC_STAT_AREA]) - edge["area"]) / max(edge["area"], 1)))
        if abs(float(centers[selected, 0]) - edge["x"]) > 2:
            raise ValueError(f"Cannot identify original proposal: {edge}")
        observed = components == selected
        repaired, added, _ = complete_component(observed, ridge_image(band))
        ys, xs = np.where(repaired > 0)
        x1, x2 = max(0, int(xs.min()) - 65), min(band.shape[1], int(xs.max()) + 66)
        raw = band[:, x1:x2]
        display = band.copy()
        display[observed] = (0, 210, 255)  # Yellow: original proposal.
        display[added > 0] = (255, 240, 0)  # Cyan: inferred continuation.
        overlay = cv2.addWeighted(band, .35, display, .65, 0)[:, x1:x2]
        identifier = f"R{index:02d}"
        for name, image in (("raw", raw), ("overlay", overlay)):
            if not cv2.imwrite(str(output / "assets" / f"{identifier}_{name}.png"), image):
                raise OSError(identifier)
        context = band.copy()
        cv2.rectangle(context, (x1, 0), (x2 - 1, band.shape[0] - 1),
                      (255, 240, 0), 2)
        cv2.imwrite(str(output / "assets" / f"{identifier}_context.png"), context)
        items.append({
            "id": identifier, "camera": edge["camera"], "band": edge["band"],
            "source": edge["source"], "filename": Path(edge["source"]).name,
            "x": edge["x"], "crop_x1": x1, "crop_x2": x2,
            "height": band.shape[0], "full_width": band.shape[1],
            "first_y": edge["first_y"], "last_y": edge["last_y"],
            "added_rows": edge["added_rows"], "ridge": edge["median_extension_ridge"],
            "thickness": edge["thickness"],
            "raw": f"assets/{identifier}_raw.png",
            "overlay": f"assets/{identifier}_overlay.png",
            "context": f"assets/{identifier}_context.png",
        })
    digest = hashlib.sha256(queue_bytes).hexdigest()
    payload = {"schema_version": 1, "queue_sha256": digest, "items": items}
    state_file = output / "decisions.json"
    if state_file.exists():
        prior = json.loads(state_file.read_text(encoding="utf-8"))
        if prior.get("queue_sha256") != digest:
            raise ValueError("Existing review decisions belong to a different queue")
    (output / "items.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("segment_label_review.html"), output / "index.html")
    return payload


class ReviewHandler(SimpleHTTPRequestHandler):
    review_lock = Lock()
    payload = None
    output_dir = None

    def state(self):
        path = self.output_dir / "decisions.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"schema_version": 1, "queue_sha256": self.payload["queue_sha256"],
                "decisions": {}}

    def json_response(self, value, status=200):
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/api/state":
            with self.review_lock:
                self.json_response(self.state())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/save":
            self.json_response({"error": "Unknown endpoint"}, 404)
            return
        origin = self.headers.get("Origin")
        permitted = {f"http://127.0.0.1:{self.server.server_port}",
                     f"http://localhost:{self.server.server_port}"}
        if origin and origin not in permitted:
            self.json_response({"error": "Origin rejected"}, 403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(size))
            items = {item["id"]: item for item in self.payload["items"]}
            identifier = value["id"]
            item = items[identifier]
            status = value.get("status", "")
            if status not in {"", "accept", "fix", "reject", "uncertain"}:
                raise ValueError("Invalid status")
            notes = str(value.get("notes", ""))[:2000]
            points = value.get("points", [])
            if not isinstance(points, list) or len(points) > 128:
                raise ValueError("Invalid points")
            checked_points = []
            for point in points:
                x, y = float(point["x"]), float(point["y"])
                if not np.isfinite([x, y]).all() or not (
                        item["crop_x1"] <= x <= item["crop_x2"] - 1 and
                        0 <= y <= item["height"] - 1):
                    raise ValueError("Point outside crop")
                checked_points.append({"x": round(x, 2), "y": round(y, 2)})
            with self.review_lock:
                state = self.state()
                state["decisions"][identifier] = {
                    "status": status, "notes": notes, "points": checked_points,
                    "source": item["source"], "camera": item["camera"],
                    "band": item["band"], "target_x": item["x"],
                }
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                                  dir=self.output_dir, delete=False,
                                                  suffix=".json") as handle:
                    json.dump(state, handle, ensure_ascii=False, indent=2)
                    temp_path = Path(handle.name)
                temp_path.replace(self.output_dir / "decisions.json")
            self.json_response({"saved": identifier})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.json_response({"error": str(error)}, 400)

    def log_message(self, format, *args):
        if args and "/api/save" in str(args[0]):
            print(format % args, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dataset", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    payload = prepare(args.point_dataset, args.curated, args.queue, output)
    print("Prepared", len(payload["items"]), "edges at", output, flush=True)
    if args.prepare_only:
        return
    ReviewHandler.payload = payload
    ReviewHandler.output_dir = output
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 partial(ReviewHandler, directory=str(output)))
    print(f"Review UI: http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
