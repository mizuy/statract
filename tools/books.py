"""Book bibliography: ``books.yaml`` in, ``manual_entries.bib`` out.

Paperpile holds the papers. Books live here, keyed by ISBN, and are checked
against the National Diet Library (NDL Search OpenSearch API), with openBD as
fallbacks (openBD for Japanese ISBNs, Open Library for others) when NDL
has no record::

    books.yaml          what the docs print (hand-written, the source)
    books.lock.json     what NDL / openBD say for each ISBN (fetched)
    manual_entries.bib  generated from books.yaml; bibliase merges it

Commands (``task books:<cmd>``)::

    python tools/books.py add KEY ISBN [ISBN ...]  # fetch, print a books.yaml entry
    python tools/books.py fetch [--refresh]        # fill books.lock.json
    python tools/books.py build                    # write manual_entries.bib
    python tools/books.py check                    # offline; CI runs this

``check`` needs no network. It fails when an ISBN has a bad check digit or no
lock record, when ``manual_entries.bib`` is out of date, or when the title,
year, publisher, edition or a person in ``books.yaml`` does not match any
record for that ISBN. A difference that is right on purpose (a shortened
title, a co-publisher) goes in the entry's ``accept`` map with the reason.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books.yaml"
LOCK = ROOT / "books.lock.json"
BIB = ROOT / "manual_entries.bib"

NDL_URL = "https://ndlsearch.ndl.go.jp/api/opensearch?isbn={isbn}"
OPENBD_URL = "https://api.openbd.jp/v1/get?isbn={isbn}"
OPENLIBRARY_URL = "https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
USER_AGENT = "statract-books/1 (+https://github.com/mizuy/statract)"
# NDL asks for no concurrent access (https://ndlsearch.ndl.go.jp/help/api).
# Requests go one at a time, each PAUSE seconds after the last answer.
PAUSE = 5.0
_last_answer = 0.0

FIELDS = (
    "title", "author", "translator", "edition", "series", "number",
    "publisher", "location", "date", "langid",
)
PEOPLE = ("author", "translator")
CHECKED = ("isbn", "title", "date", "publisher", "edition", *PEOPLE)

HEADER = """\
% Generated from books.yaml by tools/books.py (task books:build).
% Do not edit by hand. bibliase bib merges this file into references.bib.
"""

NS = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcndl": "http://ndl.go.jp/dcndl/terms/",
    "dcterms": "http://purl.org/dc/terms/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}
XSI_TYPE = f"{{{NS['xsi']}}}type"


# ---------------------------------------------------------------- ISBN

def isbn_digits(raw: object) -> str:
    return re.sub(r"[^0-9Xx]", "", str(raw)).upper()


def isbn13_ok(isbn: str) -> bool:
    if not re.fullmatch(r"97[89]\d{10}", isbn):
        return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(isbn[:12]))
    return (10 - total % 10) % 10 == int(isbn[12])


def entry_isbns(entry: dict) -> list[tuple[str | None, str]]:
    """``[(volume, isbn), ...]``. ``isbn`` is one value or a volume map."""
    raw = entry.get("isbn")
    if isinstance(raw, dict):
        return [(str(vol), isbn_digits(v)) for vol, v in raw.items()]
    if raw is None:
        return []
    return [(None, isbn_digits(raw))]


# ---------------------------------------------------------------- sources

def _get(url: str) -> bytes:
    global _last_answer
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        # Wait PAUSE after the previous answer; longer after each failure.
        wait = PAUSE * 2 ** attempt - (time.monotonic() - _last_answer)
        if wait > 0:
            time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            body = b"<error><code>%d</code></error>" % e.code
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            body = b"<error>timeout</error>"
        finally:
            _last_answer = time.monotonic()
        # NDL reports overload in the body with status 200.
        if not body.lstrip().startswith(b"<error>"):
            return body
    raise RuntimeError(f"giving up on {url}")


def parse_ndl(xml: bytes) -> list[dict]:
    """One record per ``<item>`` of an NDL OpenSearch (RSS 2.0) response."""
    root = ET.fromstring(xml)
    records = []
    for item in root.iter("item"):
        def text(tag: str) -> str | None:
            el = item.find(tag, NS)
            return el.text.strip() if el is not None and el.text else None

        ids = {}
        for el in item.findall("dc:identifier", NS):
            kind = (el.get(XSI_TYPE) or "").split(":")[-1]
            if kind in ("NDLBibID", "NIIBibID") and el.text:
                ids[kind] = el.text.strip()
        desc = item.findtext("description") or ""
        resp = re.search(r"責任表示：(.*?)</li>", desc)
        records.append({
            "title": text("dc:title"),
            "volume": text("dcndl:volume"),
            "edition": text("dcndl:edition"),
            "series": text("dcndl:seriesTitle"),
            "publisher": text("dc:publisher"),
            "date": text("dc:date"),
            "issued": text("dcterms:issued"),
            "creators": [el.text.strip() for el in item.findall("dc:creator", NS) if el.text],
            "responsibility": resp.group(1).strip() if resp else None,
            "ids": ids,
            "url": item.findtext("link"),
        })
    # Records NDL catalogued itself first; CiNii copies after.
    records.sort(key=lambda r: "NDLBibID" not in r["ids"])
    return [{k: v for k, v in r.items() if v} for r in records]


def parse_openbd(body: bytes) -> list[dict]:
    records = []
    for hit in json.loads(body):
        if not hit:
            continue
        s = hit.get("summary", {})
        records.append({k: v for k, v in {
            "title": s.get("title"),
            "volume": s.get("volume"),
            "series": s.get("series"),
            "publisher": s.get("publisher"),
            "date": (s.get("pubdate") or "")[:4] or None,
            "responsibility": s.get("author"),
        }.items() if v})
    return records


def parse_openlibrary(body: bytes) -> list[dict]:
    records = []
    for hit in json.loads(body).values():
        title = hit.get("title")
        if hit.get("subtitle"):
            title = f"{title} : {hit['subtitle']}"
        records.append({k: v for k, v in {
            "title": title,
            "publisher": ", ".join(p["name"] for p in hit.get("publishers", [])),
            "date": _year(hit.get("publish_date")),
            "creators": [a["name"] for a in hit.get("authors", [])],
            "url": hit.get("url"),
        }.items() if v})
    return records


def fetch_isbn(isbn: str) -> dict:
    records = parse_ndl(_get(NDL_URL.format(isbn=isbn)))
    source = "ndl"
    if not records:
        if isbn.startswith("9784"):
            source, url, parse = "openbd", OPENBD_URL, parse_openbd
        else:
            source, url, parse = "openlibrary", OPENLIBRARY_URL, parse_openlibrary
        records = parse(_get(url.format(isbn=isbn)))
        if not records:
            source = "none"
    return {"source": source, "fetched": _dt.date.today().isoformat(), "records": records}


# ---------------------------------------------------------------- files

def load_books(path: Path = BOOKS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def load_lock(path: Path = LOCK) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_lock(lock: dict, path: Path = LOCK) -> None:
    text = json.dumps(dict(sorted(lock.items())), ensure_ascii=False, indent=1)
    path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------- bib

_CJK = re.compile(r"[぀-ヿ㐀-鿿＀-￯]")


def _person(name: str) -> str:
    # Japanese names go in braces so biblatex/citeproc keep them whole.
    return name if name == "others" or not _CJK.search(name) else "{" + name + "}"


def _value(field: str, entry: dict) -> str | None:
    if field in PEOPLE:
        people = entry.get(field)
        if not people:
            return None
        if isinstance(people, str):
            people = [people]
        return " and ".join(_person(p) for p in people)
    v = entry.get(field)
    return None if v is None else str(v)


def to_bib(entry: dict) -> str:
    pairs = [(f, _value(f, entry)) for f in FIELDS]
    vols = entry_isbns(entry)
    isbn = ", ".join(f"{i} ({v})" if v else i for v, i in vols)
    pairs.append(("isbn", isbn or None))
    pairs = [(f, v) for f, v in pairs if v is not None]
    width = max(len(f) for f, _ in pairs)
    body = ",\n".join(f"  {f.ljust(width)} = {{{v}}}" for f, v in pairs)
    return f"@BOOK{{{entry['key']},\n{body}\n}}\n"


def build(books: list[dict]) -> str:
    return HEADER + "".join("\n" + to_bib(e) for e in books)


# ---------------------------------------------------------------- check

def norm(s: object) -> str:
    """Compare text loosely: width, case, accents, spaces, punctuation, braces."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\band\b", "&", s)
    # Drop accents (Hernán = Hernan) but keep kana voicing marks (ブ != フ).
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not (unicodedata.combining(c) and ord(c) < 0x3000)
    )
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if unicodedata.category(c)[0] not in "PZS")


def _family(name: str) -> str:
    """"Rubin, Donald B" and "Donald B. Rubin" (Open Library) -> rubin."""
    name = name.split(",")[0] if "," in name else (name.split() or [""])[-1]
    return norm(name)


def _year(s: str | None) -> str | None:
    m = re.search(r"\d{4}", s or "")
    return m.group(0) if m else None


def _edition(s: str | None) -> str:
    if not s:
        return "1"
    m = re.search(r"\d+", unicodedata.normalize("NFKC", s))
    return m.group(0) if m else norm(s)


def _people(entry: dict, field: str) -> list[str]:
    p = entry.get(field) or []
    return [p] if isinstance(p, str) else list(p)


def _person_ok(name: str, records: list[dict]) -> bool | None:
    """True/False when the name can be checked against NDL creators.

    Latin names are checked by family name, kanji names as a whole. Katakana
    renderings of foreign names ("J. パール") have no NDL form: ``None``.
    """
    if name == "others" or re.search(r"[゠-ヿ]", name):
        return None
    creators = [c for r in records for c in r.get("creators", [])]
    if not creators:
        return None
    if _CJK.search(name):
        return any(norm("".join(c.split(",")[:2])) == norm(name) for c in creators)
    return any(_family(c) == _family(name) for c in creators)


def check_entry(entry: dict, lock: dict) -> list[str]:
    key = entry.get("key", "?")
    accept = entry.get("accept") or {}
    problems: list[str] = []

    def bad(field: str, msg: str) -> None:
        if field not in accept:
            problems.append(f"{key}: {field}: {msg}")

    for field, reason in accept.items():
        if field not in CHECKED:
            problems.append(f"{key}: accept: unknown field {field!r}")
        elif not str(reason or "").strip():
            problems.append(f"{key}: accept: {field} needs a reason")

    vols = entry_isbns(entry)
    if not vols:
        problems.append(f"{key}: isbn: missing")
        return problems
    records: list[dict] = []
    for _, isbn in vols:
        if not isbn13_ok(isbn):
            problems.append(f"{key}: isbn: {isbn} is not a valid ISBN-13")
        elif isbn not in lock:
            problems.append(f"{key}: isbn: {isbn} not in books.lock.json (task books:fetch)")
        elif not lock[isbn]["records"]:
            bad("isbn", f"{isbn} not found in NDL or openBD")
        else:
            records += lock[isbn]["records"]
    if not records:
        return problems

    title = str(entry.get("title", ""))
    if len(vols) > 1:  # "書名（上・下）" covers both volumes
        title = re.sub(r"[（(][^（）()]*[）)]\s*$", "", title)
    titles = {norm(r["title"]) for r in records if r.get("title")}
    # A volume in a set: OpenSearch gives the set title and volume number
    # ("医学のための因果推論", "1") but not the volume title. Check those.
    in_set = any(
        norm(r.get("title", "")) == norm(entry.get("series", "-"))
        and norm(r.get("volume", "")) == norm(entry.get("number", "-"))
        for r in records
    )
    if norm(title) not in titles and not in_set:
        got = " | ".join(sorted({r["title"] for r in records if r.get("title")}))
        bad("title", f"{entry.get('title')!r} vs NDL {got!r}")

    years = {_year(r.get("date") or r.get("issued")) for r in records}
    if _year(str(entry.get("date"))) not in years:
        bad("date", f"{entry.get('date')} vs {sorted(y for y in years if y)}")

    # "CRC Press" matches "CRC Press, an imprint of the Taylor & Francis Group".
    pubs = {norm(r["publisher"]) for r in records if r.get("publisher")}
    pub = norm(entry.get("publisher", ""))
    if pubs and not (pub and any(pub in p for p in pubs)):
        got = " | ".join(sorted({r["publisher"] for r in records if r.get("publisher")}))
        bad("publisher", f"{entry.get('publisher')!r} vs {got!r}")

    # A copy without an edition statement does not outvote one with it.
    eds = {_edition(r["edition"]) for r in records if r.get("edition")} or {"1"}
    if _edition(entry.get("edition") and str(entry["edition"])) not in eds:
        got = sorted({r.get("edition") or "(none)" for r in records})
        bad("edition", f"{entry.get('edition')!r} vs {got}")

    for field in PEOPLE:
        for name in _people(entry, field):
            if _person_ok(name, records) is False:
                bad(field, f"{name!r} not among {records[0].get('creators')}")
    return problems


def check(books: list[dict], lock: dict, bib_text: str) -> list[str]:
    problems = []
    keys = [e.get("key") for e in books]
    for k in sorted({k for k in keys if keys.count(k) > 1}):
        problems.append(f"{k}: duplicate key")
    for e in books:
        unknown = set(e) - set(FIELDS) - {"key", "isbn", "accept", "note"}
        if unknown:
            problems.append(f"{e.get('key')}: unknown fields {sorted(unknown)}")
        problems += check_entry(e, lock)
    if bib_text != build(books):
        problems.append("manual_entries.bib is out of date (task books:build)")
    return problems


# ---------------------------------------------------------------- add

def suggest(key: str, isbns: list[str], lock: dict) -> dict:
    """A books.yaml entry filled from the first lock record."""
    rec = lock[isbns[0]]["records"][0] if lock[isbns[0]]["records"] else {}
    japanese = bool(_CJK.search(rec.get("title", "")))
    title = rec.get("title", "")
    title = title.replace(" : ", "：") if japanese else title.replace(" : ", ": ")
    # NDL: "黒木, 学, 統計科学", "繁桝, 算男, 1946-", "Rubin, Donald B".
    people = [
        "".join(c.split(",")[:2]).replace(" ", "") if _CJK.search(c)
        else re.sub(r",\s*\d{4}-\d*$", "", c)
        for c in rec.get("creators", [])
    ]
    entry = {"key": key, "title": title, "author": people or ["TODO"]}
    if rec.get("edition"):
        entry["edition"] = rec["edition"]
    if rec.get("series"):
        entry["series"] = rec["series"]
    entry["publisher"] = rec.get("publisher", "TODO")
    if japanese:
        entry["location"] = "東京"
    entry["date"] = int(_year(rec.get("date")) or 0) or "TODO"
    if japanese:
        entry["langid"] = "japanese"
    entry["isbn"] = isbns[0] if len(isbns) == 1 else {f"vol{i + 1}": v for i, v in enumerate(isbns)}
    entry["note"] = f"NDL 責任表示: {rec.get('responsibility', '?')} (split author/translator by hand)"
    return entry


# ---------------------------------------------------------------- main

def _fetch(isbns: list[str], lock: dict, refresh: bool) -> bool:
    """Fetch and save after each ISBN, so a later failure keeps the rest."""
    todo = [i for i in dict.fromkeys(isbns) if refresh or i not in lock]
    ok = True
    for isbn in todo:
        try:
            lock[isbn] = fetch_isbn(isbn)
        except Exception as e:  # noqa: BLE001 - report and go on
            print(f"{isbn}: FAILED {e}", file=sys.stderr)
            ok = False
            continue
        save_lock(lock)
        print(f"{isbn}: {lock[isbn]['source']}, {len(lock[isbn]['records'])} record(s)", file=sys.stderr)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="fetch lock records for every ISBN in books.yaml")
    p.add_argument("--refresh", action="store_true", help="fetch again even if locked")
    p.add_argument("isbn", nargs="*", help="only these ISBNs")
    p = sub.add_parser("add", help="fetch ISBNs and print a books.yaml entry")
    p.add_argument("key")
    p.add_argument("isbn", nargs="+")
    sub.add_parser("build", help="write manual_entries.bib")
    sub.add_parser("check", help="offline check (CI)")
    args = ap.parse_args(argv)

    lock = load_lock()
    if args.cmd == "fetch":
        books = load_books()
        isbns = [isbn_digits(i) for i in args.isbn] or [i for e in books for _, i in entry_isbns(e)]
        return 0 if _fetch(isbns, lock, args.refresh or bool(args.isbn)) else 1
    elif args.cmd == "add":
        isbns = [isbn_digits(i) for i in args.isbn]
        for i in isbns:
            if not isbn13_ok(i):
                ap.error(f"{i} is not a valid ISBN-13")
        if not _fetch(isbns, lock, refresh=False):
            return 1
        out = yaml.safe_dump([suggest(args.key, isbns, lock)], allow_unicode=True, sort_keys=False)
        print("# Check against the book, then append to books.yaml:\n" + out)
    elif args.cmd == "build":
        BIB.write_text(build(load_books()), encoding="utf-8")
    elif args.cmd == "check":
        problems = check(load_books(), lock, BIB.read_text(encoding="utf-8"))
        for p in problems:
            print(p)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
