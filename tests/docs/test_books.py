"""tools/books.py: books.yaml -> manual_entries.bib, checked against NDL."""

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "books.py"
_spec = importlib.util.spec_from_file_location("books", _TOOL)
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)

# Trimmed NDL OpenSearch answer for ISBN 9784254122411: a CiNii copy first,
# then the record NDL catalogued itself.
NDL_XML = """\
<rss xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcndl="http://ndl.go.jp/dcndl/terms/"
     xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     version="2.0">
  <channel>
    <item>
      <title>統計的因果推論 : 入門</title>
      <link>https://ndlsearch.ndl.go.jp/books/R000000004-I029878391</link>
      <description><![CDATA[<p>朝倉書店,2019<p><ul><li>責任表示：Judea Pearl [著] ; 落海浩訳</li></ul>]]></description>
      <dc:title>統計的因果推論 : 入門</dc:title>
      <dc:creator>Pearl, Judea</dc:creator>
      <dc:creator>落海, 浩</dc:creator>
      <dc:publisher>朝倉書店</dc:publisher>
      <dc:date xsi:type="dcterms:W3CDTF">2019</dc:date>
      <dc:identifier xsi:type="dcndl:NIIBibID">BB28798886</dc:identifier>
    </item>
    <item>
      <title>入門統計的因果推論</title>
      <link>https://ndlsearch.ndl.go.jp/books/R100000002-I029878391</link>
      <description><![CDATA[<p>朝倉書店,2019<p><ul><li>責任表示：Judea Pearl, Madelyn Glymour, Nicholas P.Jewell [著],落海浩 訳</li></ul>]]></description>
      <dc:title>入門統計的因果推論</dc:title>
      <dc:creator>Pearl, Judea</dc:creator>
      <dc:creator>Glymour, Madelyn</dc:creator>
      <dc:creator>Jewell, Nicholas P., 1952-</dc:creator>
      <dc:creator>落海, 浩, 1966-</dc:creator>
      <dcndl:edition>第2版</dcndl:edition>
      <dc:publisher>朝倉書店</dc:publisher>
      <dc:date xsi:type="dcterms:W3CDTF">2019</dc:date>
      <dcterms:issued>2019.8</dcterms:issued>
      <dc:identifier xsi:type="dcndl:ISBN">978-4-254-12241-1</dc:identifier>
      <dc:identifier xsi:type="dcndl:NDLBibID">029878391</dc:identifier>
    </item>
  </channel>
</rss>
""".encode()

ISBN = "9784254122411"


def entry(**kw):
    e = {
        "key": "Pearl2019-book-ja",
        "title": "入門 統計的因果推論",
        "author": ["J. パール"],
        "translator": ["落海浩"],
        "edition": "第 2 版",
        "publisher": "朝倉書店",
        "location": "東京",
        "date": 2019,
        "langid": "japanese",
        "isbn": int(ISBN),
    }
    e.update(kw)
    return e


@pytest.fixture
def lock():
    return {ISBN: {"source": "ndl", "fetched": "2026-10-13", "records": bk.parse_ndl(NDL_XML)}}


def test_isbn_check_digit():
    assert bk.isbn13_ok("9784254122411")
    assert not bk.isbn13_ok("9784254122412")
    assert not bk.isbn13_ok("4254122411")


def test_parse_ndl_puts_ndl_record_first():
    recs = bk.parse_ndl(NDL_XML)
    assert [r["title"] for r in recs] == ["入門統計的因果推論", "統計的因果推論 : 入門"]
    assert recs[0]["ids"] == {"NDLBibID": "029878391"}
    assert recs[0]["edition"] == "第2版"
    assert recs[0]["responsibility"].startswith("Judea Pearl, Madelyn")


def test_parse_ndl_empty():
    assert bk.parse_ndl(b'<rss version="2.0"><channel></channel></rss>') == []


def test_matching_entry_passes(lock):
    assert bk.check_entry(entry(), lock) == []


@pytest.mark.parametrize(
    "change, field",
    [
        ({"title": "統計的因果推論"}, "title"),
        ({"date": 2018}, "date"),
        ({"publisher": "共立出版"}, "publisher"),
        ({"edition": None}, "edition"),
        ({"translator": ["黒木学"]}, "translator"),
        ({"author": ["Rubin, Donald B"]}, "author"),
    ],
)
def test_mismatch_is_reported_unless_accepted(lock, change, field):
    problems = bk.check_entry(entry(**change), lock)
    assert [p.split(": ")[1] for p in problems] == [field]
    accepted = entry(**change, accept={field: "intended"})
    assert bk.check_entry(accepted, lock) == []


def test_accept_needs_reason(lock):
    problems = bk.check_entry(entry(accept={"title": ""}), lock)
    assert problems == ["Pearl2019-book-ja: accept: title needs a reason"]


def test_bad_or_unlocked_isbn():
    problems = bk.check_entry(entry(isbn=9784254122412), {})
    assert "not a valid ISBN-13" in problems[0]
    problems = bk.check_entry(entry(), {})
    assert "not in books.lock.json" in problems[0]


def test_volumes_share_one_entry(lock):
    other = "9784254122923"
    lock[other] = lock[ISBN]
    e = entry(title="入門 統計的因果推論（上・下）", isbn={"上": int(ISBN), "下": int(other)})
    assert bk.check_entry(e, lock) == []
    assert f"isbn       = {{{ISBN} (上), {other} (下)}}" in bk.to_bib(e)


def test_to_bib_braces_japanese_names_only():
    bib = bk.to_bib(entry(author=["Pearl, Judea", "J. パール"], translator=["落海浩", "others"]))
    assert "author     = {Pearl, Judea and {J. パール}}" in bib
    assert "translator = {{落海浩} and others}" in bib


def test_build_and_check_detect_stale_bib(lock):
    books = [entry()]
    assert bk.check(books, lock, bk.build(books)) == []
    assert bk.check(books, lock, "") == ["manual_entries.bib is out of date (task books:build)"]


def test_suggest_fills_from_ndl(lock):
    e = bk.suggest("Pearl2019-book-ja", [ISBN], lock)
    assert e["title"] == "入門統計的因果推論"
    assert e["author"][-1] == "落海浩"
    assert e["author"][2] == "Jewell, Nicholas P."
    assert e["date"] == 2019 and e["langid"] == "japanese"


def test_repo_books_are_consistent():
    """The committed books.yaml, lock and bib agree (same as task books:check)."""
    problems = bk.check(bk.load_books(), bk.load_lock(), bk.BIB.read_text(encoding="utf-8"))
    assert problems == []


def test_requests_wait_after_each_answer(monkeypatch):
    """NDL: no concurrent access; next request PAUSE s after the last answer."""
    clock = [1000.0]
    calls = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    answers = [b"<error><code>429</code></error>", b"<rss/>", b"<rss/>"]

    def urlopen(req, timeout):
        calls.append(clock[0])
        clock[0] += 1.0  # the server takes 1 s to answer
        return Resp(answers.pop(0))

    monkeypatch.setattr(bk.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(bk.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(bk.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(bk, "_last_answer", 0.0)
    bk._get("https://example.org/a")  # 429, then retry
    bk._get("https://example.org/b")
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    # 1 s answer + 10 s back-off after the 429; 1 s answer + 5 s pause after.
    assert gaps == [11.0, 6.0]
