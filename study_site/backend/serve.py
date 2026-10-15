"""Build and preview the site; serve the same JSON through optional local APIs."""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from build import SITE_ROOT, build


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/api/catalog":
            self.path = "/data/catalog.json"
        elif parsed.path == "/api/document":
            identifier = parse_qs(parsed.query).get("id", [""])[0]
            catalog = json.loads((Path(self.directory) / "data/catalog.json").read_text())
            document = next((doc for doc in catalog["documents"] if doc["id"] == identifier), None)
            if document is None:
                self.send_error(404, "Document not found")
                return
            self.path = "/data/documents/" + document["file"]
        super().do_GET()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(SITE_ROOT / "dist")))
    print(f"Study library: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
