"""Check generated content, translations, safe rendering, and local API routes."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import urljoin

from build import REPOSITORY_ROOT, build, render_version
from serve import Handler


class SiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="algorithm-atlas-test-")
        cls.output = Path(cls.temporary.name) / "site"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.catalog = build(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_three_collections_have_working_pages(self):
        self.assertEqual({item["id"] for item in self.catalog["collections"]}, {"ai-infra", "1point3acres", "leetgpu"})
        ids = {item["id"] for item in self.catalog["documents"]}
        self.assertEqual(len(ids), len(self.catalog["documents"]))
        for collection in self.catalog["collections"]:
            self.assertIn(collection["defaultDocument"], ids)
            self.assertGreater(collection["documents"], 0)

    def test_bilingual_answers_cover_the_same_seven_questions(self):
        document = next(
            doc
            for doc in self.catalog["documents"]
            if doc["relative"]
            == "AIInfraGuide_Interview/comprehensive/general/ai-infra-first-round/README.md"
        )
        data = json.loads((self.output / "data/documents" / document["file"]).read_text())
        self.assertEqual(set(data["versions"]), {"en", "zh"})
        english, chinese = data["versions"]["en"], data["versions"]["zh"]
        self.assertEqual(len(english["sections"]), 7)
        self.assertEqual(len(chinese["sections"]), 7)
        self.assertNotEqual(english["title"], chinese["title"])
        self.assertIn("基本原理", chinese["sections"][0]["title"])
        self.assertNotIn("{{python-block-", json.dumps(data))
        self.assertEqual([section["id"] for section in english["sections"]], [section["id"] for section in chinese["sections"]])

    def test_downloads_match_repository_sources(self):
        for document in self.catalog["documents"]:
            payload = self.output / "data/documents" / document["file"]
            self.assertTrue(payload.is_file())
            downloaded = self.output / "sources" / document["relative"]
            self.assertEqual(downloaded.read_bytes(), (REPOSITORY_ROOT / document["relative"]).read_bytes())

    def test_tree_contains_every_document(self):
        found = set()

        def visit(node):
            if "documentId" in node:
                found.add(node["documentId"])
            for child in node["children"]:
                visit(child)

        for collection in self.catalog["collections"]:
            visit(collection["tree"])
        self.assertEqual(found, {doc["id"] for doc in self.catalog["documents"]})

    def test_only_configured_content_is_published(self):
        published = [path.relative_to(self.output).as_posix() for path in self.output.rglob("*")]
        self.assertFalse(any(".skill-fix" in path or ".git/" in path or ".venv" in path for path in published))
        for name in ("index.html", "app.js", "styles.css", "favicon.svg", "data/catalog.json", ".nojekyll"):
            self.assertTrue((self.output / name).is_file(), name)

    def test_markdown_does_not_execute_embedded_html(self):
        value = render_version('# Test\n\n<script>alert(1)</script>\n\n[x](javascript:alert)\n\n<img src="x" onerror="alert(1)">', "Test", REPOSITORY_ROOT / "README.md", {})
        self.assertNotIn("<script", value["intro"])
        self.assertNotIn("javascript:", value["intro"])
        self.assertNotIn("onerror=", value["intro"])

    def test_assignment_page_excludes_the_separate_test_runner(self):
        relatives = {document["relative"] for document in self.catalog["documents"]}
        assignment = (
            "AIInfraGuide_Interview/comprehensive/general/ai-infra-first-round/"
            "7.grouped_query_attention/"
            "7.grouped_query_attention.py"
        )
        test_runner = (
            "AIInfraGuide_Interview/comprehensive/general/ai-infra-first-round/"
            "7.grouped_query_attention/"
            "7.grouped_query_attention_test.py"
        )
        self.assertIn(assignment, relatives)
        self.assertNotIn(test_runner, relatives)

    def test_static_files_and_local_api(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(self.output)))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/") as response:
                self.assertIn(b"Algorithm Atlas", response.read())
            with urlopen(base + "/api/catalog") as response:
                self.assertEqual(json.load(response)["schema"], 1)
            document = self.catalog["documents"][0]
            with urlopen(base + "/api/document?id=" + document["id"]) as response:
                self.assertEqual(json.load(response)["id"], document["id"])
            with self.assertRaises(HTTPError) as result:
                urlopen(base + "/api/document?id=missing")
            self.assertEqual(result.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_project_site_subpath_assets_and_deep_link(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(self.output.parent)))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}/site/"
        try:
            for relative in ("", "styles.css", "app.js", "favicon.svg", "data/catalog.json"):
                with urlopen(urljoin(base, relative)) as response:
                    self.assertEqual(response.status, 200)
            with urlopen(base + "#collection=ai-infra&question=question-5") as response:
                self.assertIn(b"Algorithm Atlas", response.read())
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_rebuild_removes_stale_generated_content(self):
        with tempfile.TemporaryDirectory(prefix="algorithm-atlas-rebuild-") as temporary:
            output = Path(temporary) / "site"
            with contextlib.redirect_stdout(io.StringIO()):
                build(output)
                stale = output / "data/documents/old.json"
                stale.write_text('{"outdated":true}')
                build(output)
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
