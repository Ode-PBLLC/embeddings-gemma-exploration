"""Offline tests for the Wikipedia ingester: chunking, URLs, article ordering."""

from gallery.ingest import nasa, wiki
from gallery.schema import WIKI_COLOR


def test_chunk_keeps_long_prose_paragraphs_with_sections(fixture_text):
    chunks = wiki.chunk_article(fixture_text("wiki_extract.txt"))
    sections = [s for s, _ in chunks]
    assert sections == [None, "Early life", "Education", "Legacy"]
    assert all(len(p.split()) >= 40 for _, p in chunks)


def test_chunk_skips_reference_sections_and_their_subsections(fixture_text):
    text = fixture_text("wiki_extract.txt")
    ref_lines = [l for l in text.splitlines() if l.startswith(("Wildenstein", "Another long line"))]
    assert all(len(l.split()) >= 40 for l in ref_lines)  # length alone would keep them
    paras = " ".join(p for _, p in wiki.chunk_article(text))
    assert "Wildenstein" not in paras and "Another long line" not in paras


def test_chunk_caps_paragraphs_per_article():
    para = " ".join(["word"] * 50)
    text = "\n".join([para] * 10)
    assert len(wiki.chunk_article(text)) == 6
    assert len(wiki.chunk_article(text, max_paras=3)) == 3


def test_chunk_drops_math():
    para = "The flux is {\\displaystyle F} " + " ".join(["word"] * 50)
    assert wiki.chunk_article(para) == []


def test_article_url_with_section_anchor():
    assert wiki.article_url("Claude Monet", None) == "https://en.wikipedia.org/wiki/Claude_Monet"
    assert wiki.article_url("Claude Monet", "Early life") == "https://en.wikipedia.org/wiki/Claude_Monet#Early_life"
    assert wiki.article_url("Cassini–Huygens", None) == "https://en.wikipedia.org/wiki/Cassini%E2%80%93Huygens"


def test_rows_for_page(fixture_text):
    page = {"pageid": 46406, "title": "Claude Monet", "extract": fixture_text("wiki_extract.txt")}
    rows = wiki.rows_for(page)
    assert [r["id"] for r in rows] == ["wiki:46406:0", "wiki:46406:1", "wiki:46406:2", "wiki:46406:3"]
    assert rows[0]["title"] == "Claude Monet"
    assert rows[1]["title"] == "Claude Monet — Early life"
    assert rows[1]["source_url"].endswith("#Early_life")
    assert all(r["color_hex"] == WIKI_COLOR and r["modality"] == "text" and r["media_path"] == "" for r in rows)
    assert all(r["license"] == "CC BY-SA 4.0" and r["topic"] == "Claude Monet" for r in rows)


def test_interleave_alternates_artists_and_styles():
    order = wiki.interleave(["A1", "A2", "A3"], ["S1"])
    assert order == [("artist", "A1"), ("style", "S1"), ("artist", "A2"), ("artist", "A3")]


def test_style_variants():
    assert wiki.style_variants("Edo (japanese period)") == ["Edo (japanese period)", "Edo art"]
    assert wiki.style_variants("Yoruba") == ["Yoruba", "Yoruba art"]


def test_main_balances_art_and_space(monkeypatch, tmp_path):
    long = " ".join(["word"] * 45)
    # art articles yield 1 paragraph, space articles 3: balance must come from drawing more art
    pages = {f"A{i}": (i, 1) for i in range(40)} | {f"X{i}": (100 + i, 3) for i in range(40)}
    monkeypatch.setattr(wiki, "aic_titles", lambda: ([f"A{i}" for i in range(40)], []))
    monkeypatch.setattr(wiki, "space_titles", lambda: [f"X{i}" for i in range(40)])
    monkeypatch.setattr(wiki, "resolve", lambda t: {"pageid": pages[t][0], "title": t,
                                                    "extract": "\n".join([long] * pages[t][1])})
    written = {}
    monkeypatch.setattr(wiki.C, "write_parquet", lambda df, s: written.setdefault("df", df) is None or tmp_path)
    wiki.main(["--target", "30"])
    df = written["df"]
    assert len(df) == 30
    art = df["topic"].str.startswith("A").sum()
    assert 13 <= art <= 17


def test_every_nasa_keyword_has_wiki_subjects():
    assert set(wiki.SPACE_TOPICS) == set(nasa.KEYWORDS)
    titles = wiki.space_titles()
    assert len(titles) == len(set(titles))
    assert titles[: len(nasa.KEYWORDS)] == [wiki.SPACE_TOPICS[k][0] for k in nasa.KEYWORDS]


def test_resolve_skips_missing_and_disambiguation(monkeypatch):
    pages = {
        "Gone": {"query": {"pages": [{"title": "Gone", "missing": True}]}},
        "Mercury": {"query": {"pages": [{"pageid": 1, "title": "Mercury", "pageprops": {"disambiguation": ""}}]}},
        "Jupiter": {"query": {"pages": [{"pageid": 38930, "title": "Jupiter", "extract": "x"}]}},
    }
    monkeypatch.setattr(wiki, "fetch_article", lambda t: pages[t])
    assert wiki.resolve("Gone") is None
    assert wiki.resolve("Mercury") is None
    assert wiki.resolve("Jupiter")["pageid"] == 38930
