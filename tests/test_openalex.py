# tests/test_openalex.py
import unittest
import os
import requests
from paper_find_mcp.academic_platforms.openalex import OpenAlexSearcher


def check_api_accessible():
    """Check if OpenAlex search is currently usable.

    Since 2026-02-13 OpenAlex requires a free API key and heavily rate-limits
    anonymous search. Network tests run only when a key is set AND a probe
    request returns real results; otherwise they skip (not fail).
    """
    api_key = os.environ.get("OPENALEX_API_KEY", "")
    if not api_key:
        return False
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": "test", "per_page": 1, "api_key": api_key},
            timeout=8,
        )
        return r.status_code == 200 and "results" in r.json()
    except Exception:
        return False


class TestOpenAlexOffline(unittest.TestCase):
    """Unit tests that require no network — always run."""

    def setUp(self):
        self.searcher = OpenAlexSearcher()

    def test_year_filter_parsing(self):
        p = OpenAlexSearcher._parse_year_filter
        self.assertEqual(p("2023"), "2023")
        self.assertEqual(p("2020-2023"), "2020-2023")
        self.assertEqual(p("2020-"), ">2019")   # from 2020 onward
        self.assertEqual(p("-2019"), "<2020")   # up to 2019
        self.assertIsNone(p(""))
        self.assertIsNone(p("notayear"))

    def test_abstract_reconstruction(self):
        ii = {"Returns": [0], "to": [1], "schouling": [2]}  # order by position
        self.assertEqual(
            self.searcher.reconstruct_abstract(ii), "Returns to schouling"
        )
        self.assertEqual(self.searcher.reconstruct_abstract(None), "")
        self.assertEqual(self.searcher.reconstruct_abstract({}), "")

    def test_parse_item_minimal(self):
        """_parse_openalex_item builds a valid Paper from a minimal item."""
        item = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1234/abc",
            "title": "A Test Work",
            "publication_year": 2021,
            "publication_date": "2021-06-15",
            "cited_by_count": 7,
            "type": "article",
            "authorships": [
                {"author": {"display_name": "Jane Doe"}},
                {"author": {"display_name": "John Roe"}},
            ],
            "abstract_inverted_index": {"Hello": [0], "world": [1]},
            "best_oa_location": {"pdf_url": "https://x.org/a.pdf"},
            "open_access": {"is_oa": True, "oa_url": "https://x.org/a"},
            "primary_location": {"source": {"display_name": "NBER Working Papers"}},
        }
        paper = self.searcher._parse_openalex_item(item)
        self.assertEqual(paper.doi, "10.1234/abc")
        self.assertEqual(paper.paper_id, "10.1234/abc")
        self.assertEqual(paper.title, "A Test Work")
        self.assertEqual(paper.authors, ["Jane Doe", "John Roe"])
        self.assertEqual(paper.abstract, "Hello world")
        self.assertEqual(paper.published_date.year, 2021)
        self.assertEqual(paper.citations, 7)
        self.assertEqual(paper.pdf_url, "https://x.org/a.pdf")
        self.assertEqual(paper.source, "openalex")
        self.assertEqual(paper.extra.get("container_title"), "NBER Working Papers")

    def test_empty_query(self):
        self.assertEqual(self.searcher.search("", max_results=5), [])

    def test_download_pdf_not_supported(self):
        with self.assertRaises(NotImplementedError):
            self.searcher.download_pdf("10.1234/abc", "./downloads")

    def test_read_paper_message(self):
        msg = self.searcher.read_paper("10.1234/abc", "./downloads")
        self.assertIn("OpenAlex", msg)

    def test_user_agent_header(self):
        ua = self.searcher.session.headers.get("User-Agent", "")
        self.assertIn("paper_find_mcp", ua)


class TestOpenAlexNetwork(unittest.TestCase):
    """Live API tests — skipped unless OPENALEX_API_KEY is set and usable."""

    @classmethod
    def setUpClass(cls):
        cls.api_accessible = check_api_accessible()
        if not cls.api_accessible:
            print(
                "\nWarning: OpenAlex search not accessible "
                "(no OPENALEX_API_KEY or rate-limited); network tests skipped"
            )

    def setUp(self):
        self.searcher = OpenAlexSearcher()

    def test_search(self):
        if not self.api_accessible:
            self.skipTest("OpenAlex search not accessible")
        papers = self.searcher.search(
            "english medium instruction earnings", max_results=5
        )
        self.assertGreater(len(papers), 0)
        self.assertTrue(papers[0].title)

    def test_working_paper_coverage(self):
        """Confirms RePEc's unique value (NBER/IZA/Fed WPs) is covered."""
        if not self.api_accessible:
            self.skipTest("OpenAlex search not accessible")
        papers = self.searcher.search("monetary policy", max_results=25)
        sources = " ".join(
            (p.extra or {}).get("container_title", "").lower() for p in papers
        )
        self.assertTrue(
            any(k in sources for k in ("nber", "national bureau", "iza", "federal reserve")),
            "Expected at least one NBER/IZA/Fed working paper in results",
        )

    def test_year_filter(self):
        if not self.api_accessible:
            self.skipTest("OpenAlex search not accessible")
        papers = self.searcher.search("inflation", max_results=5, year="2015-2025")
        for p in papers:
            if p.published_date:
                self.assertGreaterEqual(p.published_date.year, 2015)
                self.assertLessEqual(p.published_date.year, 2025)


if __name__ == "__main__":
    unittest.main()
