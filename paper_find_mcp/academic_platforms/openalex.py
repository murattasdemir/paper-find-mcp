# paper_find_mcp/academic_platforms/openalex.py
"""
OpenAlexSearcher - OpenAlex 学术论文搜索

Searches OpenAlex (https://openalex.org), an open catalog of 250M+ scholarly
works across all disciplines. Strong coverage of economics working papers
(NBER, IZA, Federal Reserve, CEPR) plus published journal articles, making it
a robust replacement for RePEc/IDEAS discovery without the IP-throttling.

Authentication (changed 2026-02-13):
- OpenAlex now REQUIRES an API key. Keys are free and do NOT expire; each key
  has a daily budget that refills at midnight UTC (~1,000 searches/day free).
- Create a key at https://openalex.org/settings/api and set OPENALEX_API_KEY.
- The key is passed as the `api_key` query parameter.
- `mailto` (polite pool) is still sent when available as good etiquette.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
import time
import os
import logging

from ..paper import Paper

logger = logging.getLogger(__name__)


class PaperSource:
    """Abstract base class for paper sources"""
    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError


class OpenAlexSearcher(PaperSource):
    """OpenAlex 学术论文搜索器

    使用 OpenAlex REST API 搜索学术论文元数据。覆盖全学科，
    尤其适合经济学工作论文 (NBER, IZA, Fed) 与期刊文章。

    环境变量：
    - OPENALEX_API_KEY: 免费 API key（自 2026-02-13 起必需）
    - OPENALEX_MAILTO / CROSSREF_MAILTO: 联系邮箱（polite pool，可选）
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        api_key: Optional[str] = None,
        mailto: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """初始化 OpenAlex 搜索器

        Args:
            api_key: OpenAlex API key（默认从 OPENALEX_API_KEY 环境变量获取）
            mailto: 联系邮箱（默认从 OPENALEX_MAILTO / CROSSREF_MAILTO 获取）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key or os.environ.get('OPENALEX_API_KEY', '')
        self.mailto = (
            mailto
            or os.environ.get('OPENALEX_MAILTO', '')
            or os.environ.get('CROSSREF_MAILTO', '')
        )
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'paper_find_mcp/1.0 (mailto:{self.mailto})',
            'Accept': 'application/json',
        })

        if not self.api_key:
            logger.warning(
                "No OPENALEX_API_KEY set. Since 2026-02-13 OpenAlex requires a "
                "free API key for reliable access; anonymous requests are "
                "heavily rate-limited. Create one at "
                "https://openalex.org/settings/api"
            )

    @staticmethod
    def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """从 OpenAlex 的倒排索引重建摘要文本

        OpenAlex 以 abstract_inverted_index（词 -> 位置列表）存储摘要以节省空间。
        此方法按位置还原为可读文本。

        Args:
            inverted_index: OpenAlex 的 abstract_inverted_index 字段

        Returns:
            重建的摘要字符串（无摘要时返回空串）
        """
        if not inverted_index:
            return ""
        positioned = []
        for word, positions in inverted_index.items():
            for pos in positions:
                positioned.append((pos, word))
        positioned.sort(key=lambda x: x[0])
        return " ".join(word for _, word in positioned)

    @staticmethod
    def _parse_year_filter(year: str) -> Optional[str]:
        """将年份表达式转换为 OpenAlex publication_year 过滤片段

        支持与 Semantic Scholar 相同的语法:
        '2023' -> '2023'; '2020-2023' -> '2020-2023';
        '2020-' -> '>2019'; '-2019' -> '<2020'

        Returns:
            OpenAlex filter 的 publication_year 值，无法解析时返回 None
        """
        if not year:
            return None
        year = year.strip()
        try:
            if '-' not in year:
                int(year)
                return year
            start, end = year.split('-', 1)
            start, end = start.strip(), end.strip()
            if start and end:
                return f"{int(start)}-{int(end)}"
            if start and not end:      # '2020-' => from 2020 onward
                return f">{int(start) - 1}"
            if end and not start:      # '-2019' => up to 2019
                return f"<{int(end) + 1}"
        except (ValueError, TypeError):
            logger.warning(f"Could not parse year filter: {year!r}")
        return None

    def _make_request(
        self,
        url: str,
        params: dict,
        retry_count: int = 0,
    ) -> Optional[requests.Response]:
        """发送请求，带指数退避重试"""
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)

            if response.status_code == 429:
                if retry_count < self.max_retries:
                    wait_time = (2 ** retry_count) + 1
                    logger.warning(f"Rate limited (429), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    return self._make_request(url, params, retry_count + 1)
                logger.error(f"Rate limited after {self.max_retries} retries")
                return None

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                return self._make_request(url, params, retry_count + 1)
            logger.error(f"Request failed: {e}")
            return None

    def search(
        self,
        query: str,
        max_results: int = 10,
        year: Optional[str] = None,
        **kwargs,
    ) -> List[Paper]:
        """搜索 OpenAlex 论文

        Args:
            query: 搜索关键词
            max_results: 最大返回数量（默认 10）
            year: 可选年份过滤: '2023', '2020-2023', '2020-', '-2019'

        Returns:
            List[Paper]: 论文列表（失败时返回空列表）
        """
        if not query or not query.strip():
            return []

        try:
            params = {
                'search': query,
                'per_page': min(max_results, 200),  # OpenAlex per_page max is 200
            }

            filters = []
            year_filter = self._parse_year_filter(year) if year else None
            if year_filter:
                filters.append(f"publication_year:{year_filter}")
            if filters:
                params['filter'] = ','.join(filters)

            if self.api_key:
                params['api_key'] = self.api_key
            if self.mailto:
                params['mailto'] = self.mailto

            url = f"{self.BASE_URL}/works"
            response = self._make_request(url, params)
            if not response:
                return []

            data = response.json()
            items = data.get('results', [])

            papers = []
            for item in items:
                try:
                    paper = self._parse_openalex_item(item)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing OpenAlex item: {e}")
                    continue

            return papers[:max_results]

        except requests.RequestException as e:
            logger.error(f"Error searching OpenAlex: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in OpenAlex search: {e}")
            return []

    def _parse_openalex_item(self, item: Dict[str, Any]) -> Optional[Paper]:
        """将 OpenAlex API 条目解析为 Paper 对象"""
        try:
            # DOI: OpenAlex 返回完整 URL 形式 (https://doi.org/10.x/...)，剥离为裸 DOI
            doi_url = item.get('doi') or ''
            doi = doi_url.replace('https://doi.org/', '').replace('http://doi.org/', '')

            # OpenAlex work id, e.g. https://openalex.org/W2741809807 -> W2741809807
            openalex_id = (item.get('id') or '').rstrip('/').split('/')[-1]
            paper_id = doi or openalex_id

            title = item.get('title') or item.get('display_name') or ''

            authors = [
                a['author']['display_name']
                for a in item.get('authorships', [])
                if a.get('author') and a['author'].get('display_name')
            ]

            abstract = self.reconstruct_abstract(item.get('abstract_inverted_index'))

            # 发表日期：优先 publication_date，回退 publication_year
            published_date: Optional[datetime] = None
            date_str = item.get('publication_date')
            if date_str:
                try:
                    published_date = datetime.strptime(date_str, '%Y-%m-%d')
                except (ValueError, TypeError):
                    published_date = None
            if not published_date and item.get('publication_year'):
                try:
                    published_date = datetime(int(item['publication_year']), 1, 1)
                except (ValueError, TypeError):
                    published_date = None

            # PDF / landing URL: 优先开放获取链接，回退 DOI，最后 OpenAlex 页面
            best_oa = item.get('best_oa_location') or {}
            open_access = item.get('open_access') or {}
            pdf_url = best_oa.get('pdf_url') or open_access.get('oa_url') or ''
            url = (
                doi_url
                or (best_oa.get('landing_page_url') if best_oa else '')
                or item.get('id')
                or ''
            )

            # 期刊 / 系列名称（NBER、IZA 等作为 source 出现在这里）
            primary_location = item.get('primary_location') or {}
            source = primary_location.get('source') or {}
            container_title = source.get('display_name', '') if source else ''

            return Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                doi=doi,
                published_date=published_date,
                pdf_url=pdf_url,
                url=url,
                source='openalex',
                categories=[item.get('type', '')] if item.get('type') else [],
                keywords=[
                    c['display_name']
                    for c in item.get('concepts', [])[:8]
                    if c.get('display_name')
                ],
                citations=item.get('cited_by_count', 0) or 0,
                extra={
                    'openalex_id': openalex_id,
                    'container_title': container_title,
                    'is_oa': open_access.get('is_oa', False),
                    'type': item.get('type', ''),
                },
            )

        except Exception as e:
            logger.error(f"Error parsing OpenAlex item: {e}")
            return None

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        """OpenAlex 不托管 PDF，返回替代方案说明"""
        raise NotImplementedError(
            "OpenAlex is a metadata catalog and does not host PDFs directly. "
            "Use the paper's open-access url/pdf_url when available, or "
            "download_scihub(doi) for published papers."
        )

    def read_paper(self, paper_id: str, save_path: str) -> str:
        """OpenAlex 不支持直接阅读全文，返回替代方案说明"""
        return (
            "OpenAlex papers cannot be read directly. Only metadata and "
            "reconstructed abstracts are available through the OpenAlex API. "
            "Use the open-access url when available, or read_scihub_paper(doi)."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    searcher = OpenAlexSearcher()

    print("=" * 60)
    print("Testing OpenAlex search...")
    print("=" * 60)
    papers = searcher.search("english medium instruction earnings returns", max_results=5)
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title[:70]}")
        print(f"   DOI: {paper.doi}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Year: {paper.published_date.year if paper.published_date else 'N/A'}")
        print(f"   Citations: {paper.citations}")

    print("\n" + "=" * 60)
    print("Testing year filter (2015-2025)...")
    print("=" * 60)
    recent = searcher.search("monetary policy", max_results=3, year="2015-2025")
    for paper in recent:
        yr = paper.published_date.year if paper.published_date else 'N/A'
        print(f"  [{yr}] {paper.title[:60]}")
