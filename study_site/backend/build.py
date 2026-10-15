"""Build the static study library from an explicit set of repository folders."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

import bleach
import markdown

SITE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SITE_ROOT.parent

ALLOWED_TAGS = {
    "a", "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol",
    "li", "strong", "em", "code", "pre", "blockquote", "table", "thead", "tbody",
    "tr", "th", "td", "div", "span", "sup", "sub", "del", "img",
}
ALLOWED_ATTRIBUTES = {
    "*": ["class", "id"], "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "title"], "ol": ["start"], "th": ["align"],
    "td": ["align"],
}


def natural_key(value: str) -> list:
    return [int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", value)]


def document_id(collection: dict, path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT / collection["directory"])
    suffixless = relative.parent if path.name == "README.md" else relative.with_suffix("")
    return f"{collection['id']}/{suffixless.as_posix()}".rstrip("/.")


def clean_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()


def first_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def load_markdown(path: Path) -> tuple[str, str]:
    if path.suffix == ".md":
        text = path.read_text(encoding="utf-8")
        return first_title(text, path.parent.name), text
    source = path.read_text(encoding="utf-8")
    try:
        description = ast.get_docstring(ast.parse(source)) or ""
    except SyntaxError:
        description = ""
    title = description.splitlines()[0].rstrip(".") if description else path.stem.replace("_", " ")
    fence = "`" * max(3, max((len(m.group()) + 1 for m in re.finditer(r"`+", source)), default=3))
    text = (
        f"# {title}\n\n"
        "This practice file is shown directly from the repository. Read the assignment "
        "and implementation below, or download the Python source to run it locally.\n\n"
        f"{fence}python\n{source}\n{fence}\n"
    )
    return title, text


def expand_code_includes(translation: str, english: str) -> str:
    blocks = re.findall(r"^```python\n(.*?)^```", english, re.MULTILINE | re.DOTALL)
    for index, block in enumerate(blocks, 1):
        translation = translation.replace(f"{{{{python-block-{index}}}}}", f"```python\n{block}```")
    if re.search(r"\{\{python-block-\d+\}\}", translation):
        raise ValueError("A translation references a missing English Python block")
    return translation


def render_version(text: str, title: str, source_path: Path, link_map: dict) -> dict:
    renderer = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists", "codehilite"])
    rendered = renderer.convert(text)

    def rewrite_link(match: re.Match) -> str:
        before, destination, after = match.groups()
        destination = html.unescape(destination)
        parsed = urlsplit(destination)
        attributes = ""
        if parsed.scheme in {"http", "https"}:
            attributes = ' target="_blank" rel="noopener noreferrer"'
        elif not parsed.scheme and not destination.startswith("#"):
            resolved = (source_path.parent / parsed.path).resolve()
            if resolved.is_dir():
                resolved = resolved / "README.md"
            if resolved in link_map:
                doc = link_map[resolved]
                destination = "#" + urlencode({"collection": doc["collection"], "doc": doc["id"]})
            else:
                try:
                    relative = resolved.relative_to(REPOSITORY_ROOT)
                    destination = "sources/" + quote(relative.as_posix())
                except ValueError:
                    destination = "#"
        return f'<a{before}href="{html.escape(destination, quote=True)}"{after}{attributes}>'

    rendered = re.sub(r'<a\b([^>]*?)href="([^"]*)"([^>]*)>', rewrite_link, rendered)
    rendered = bleach.clean(rendered, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, protocols={"http", "https", "mailto"}, strip=True)
    rendered = re.sub(r"<h1\b[^>]*>.*?</h1>", "", rendered, count=1, flags=re.DOTALL)
    parts = re.split(r"(?=<h2\b)", rendered)
    sections = []
    for index, section in enumerate(parts[1:], 1):
        match = re.match(r"<h2\b[^>]*>(.*?)</h2>\s*(.*)", section, flags=re.DOTALL)
        if match:
            sections.append({"id": f"question-{index}", "title": clean_text(match.group(1)), "html": match.group(2)})
    words = len(re.findall(r"\w+", text))
    return {
        "title": title, "intro": parts[0].strip(), "sections": sections,
        "minutes": max(1, round(words / 220)),
    }


def build_tree(collection: dict, documents: list[dict], titles: dict) -> dict:
    root = {"name": collection["directory"], "title": {"en": collection["title"], "zh": collection["title"]}, "children": []}
    for document in documents:
        path = Path(document["relative"])
        parts = list(path.parts[1:])
        if parts[-1] == "README.md":
            parts.pop()
        node = root
        for part in parts:
            child = next((child for child in node["children"] if child["name"] == part), None)
            if child is None:
                default = Path(part).stem.replace("_", " ") if "." in part else part.replace("-", " ")
                child = {"name": part, "title": titles.get(part, {"en": default, "zh": default}), "children": []}
                node["children"].append(child)
            node = child
        node["documentId"] = document["id"]
        if path.suffix == ".py":
            node["title"] = {"en": path.name, "zh": path.name}

    def sort(node: dict) -> None:
        node["children"].sort(key=lambda child: (not bool(child["children"]), natural_key(child["name"])))
        for child in node["children"]:
            sort(child)

    sort(root)
    return root


def build(output: Path | None = None) -> dict:
    output = (output or SITE_ROOT / "dist").resolve()
    if output == REPOSITORY_ROOT or output == SITE_ROOT or REPOSITORY_ROOT.is_relative_to(output):
        raise ValueError("Build output must be a dedicated subdirectory")
    config = json.loads((SITE_ROOT / "collections.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".atlas-generated"
    if marker.is_file():
        # Remove only directories created by a previous build of this generator.
        for name in ("data", "sources"):
            generated = output / name
            if generated.is_dir() and not generated.is_symlink():
                shutil.rmtree(generated)
    marker.write_text("Generated by Algorithm Atlas.\n", encoding="utf-8")
    (output / "data/documents").mkdir(parents=True, exist_ok=True)
    for source in (SITE_ROOT / "frontend").iterdir():
        if source.is_file():
            shutil.copy2(source, output / source.name)
    (output / ".nojekyll").touch()

    documents, link_map, inputs, indexes = [], {}, [], []
    for collection in config["collections"]:
        collection_root = (REPOSITORY_ROOT / collection["directory"]).resolve()
        collection_root.relative_to(REPOSITORY_ROOT)
        for source in sorted(collection_root.rglob("*"), key=lambda path: natural_key(path.as_posix())):
            if not source.is_file() or source.is_symlink() or source.suffix not in {".md", ".py"}:
                continue
            if source.suffix == ".py" and source.name.endswith("_test.py"):
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in source.relative_to(collection_root).parts):
                continue
            source.resolve().relative_to(collection_root)
            title, text = load_markdown(source)
            relative = source.relative_to(REPOSITORY_ROOT).as_posix()
            # Short README files with no sections are navigation, not separate lessons.
            if source.name == "README.md" and not re.search(r"^##\s", text, re.MULTILINE):
                indexes.append(source)
                continue
            identifier = document_id(collection, source)
            entry = {
                "id": identifier, "collection": collection["id"], "relative": relative,
                "kind": "questions" if source.suffix == ".md" else "practice",
                "file": hashlib.sha256(identifier.encode()).hexdigest()[:16] + ".json",
            }
            inputs.append((source, title, text, entry))
            link_map[source.resolve()] = entry
            documents.append(entry)

    for source in indexes:
        relative = source.relative_to(REPOSITORY_ROOT)
        descendant = next((doc for doc in documents if Path(doc["relative"]).is_relative_to(relative.parent)), None)
        if descendant:
            link_map[source.resolve()] = descendant
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for source, title, text, entry in inputs:
        versions = {"en": render_version(text, title, source, link_map)}
        translation_path = config["translations"].get(entry["relative"])
        if translation_path:
            translated = (SITE_ROOT / translation_path).read_text(encoding="utf-8")
            translated = expand_code_includes(translated, text)
            versions["zh"] = render_version(translated, first_title(translated, title), source, link_map)
            if len(versions["en"]["sections"]) != len(versions["zh"]["sections"]):
                raise ValueError(f"Translation question count differs: {entry['relative']}")
        entry.update({
            "title": {language: value["title"] for language, value in versions.items()},
            "languages": list(versions), "questions": len(versions["en"]["sections"]),
            "minutes": versions["en"]["minutes"],
            "outline": {language: [{"id": section["id"], "title": section["title"]} for section in value["sections"]] for language, value in versions.items()},
            "sourceUrl": f"{config['repository']}/blob/{config['branch']}/{quote(entry['relative'])}",
            "downloadUrl": "sources/" + quote(entry["relative"]),
        })
        payload = {"id": entry["id"], "versions": versions}
        (output / "data/documents" / entry["file"]).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        destination = output / "sources" / entry["relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    collections = []
    for collection in config["collections"]:
        members = [doc for doc in documents if doc["collection"] == collection["id"]]
        collections.append({
            **collection, "documents": len(members), "questions": sum(doc["questions"] for doc in members),
            "defaultDocument": members[0]["id"] if members else None,
            "tree": build_tree(collection, members, config["directory_titles"]),
        })
    catalog = {"schema": 1, "repository": config["repository"], "branch": config["branch"], "collections": collections, "documents": documents}
    (output / "data/catalog.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(collections)} collections, {len(documents)} documents, and {sum(doc['questions'] for doc in documents)} questions into {output}")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.output)
