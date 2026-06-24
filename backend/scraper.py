"""
WebIntel AI — Website Scraper

Strategy:
  1. Trafilatura (primary) — excellent at extracting main article content
  2. BeautifulSoup (fallback) — raw HTML parsing for metadata and links
  3. Combined output: title, description, clean text content, links
"""

import hashlib
import logging
from urllib.parse import urlparse, urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Maximum content length to send to LLM (roughly 15k tokens)
MAX_CONTENT_LENGTH = 50000


async def scrape_url(url: str) -> dict:
    """
    Scrape a website URL and extract structured content.

    Returns:
        {
            "url": str,
            "domain": str,
            "title": str,
            "description": str,
            "content": str,          # Clean extracted text
            "links": list[str],      # Internal/external links found
            "content_hash": str,     # MD5 hash for caching
            "success": bool,
            "error": str | None
        }
    """
    result = {
        "url": url,
        "domain": urlparse(url).netloc,
        "title": "",
        "description": "",
        "content": "",
        "links": [],
        "content_hash": "",
        "success": False,
        "error": None
    }

    try:
        # Fetch raw HTML
        html = await _fetch_html(url)
        if not html:
            result["error"] = "Failed to fetch the website. Please check the URL."
            return result

        # Extract metadata with BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        result["title"] = _extract_title(soup, url)
        result["description"] = _extract_description(soup)
        result["links"] = _extract_links(soup, url)

        # Extract main content with Trafilatura
        content = trafilatura.extract(
            html,
            include_links=True,
            include_tables=True,
            include_comments=False,
            favor_recall=True,
        )

        # Fallback: use BeautifulSoup text if Trafilatura returns nothing
        if not content or len(content.strip()) < 100:
            content = _fallback_extract(soup)

        if not content or len(content.strip()) < 50:
            result["error"] = "Could not extract meaningful content from this website."
            return result

        # Truncate to fit LLM context window
        content = content[:MAX_CONTENT_LENGTH]

        result["content"] = content
        result["content_hash"] = hashlib.md5(content.encode()).hexdigest()
        result["success"] = True

    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        result["error"] = f"Scraping error: {str(e)}"

    return result


async def _fetch_html(url: str) -> str | None:
    """Fetch raw HTML from a URL with timeout and headers."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            verify=False  # Some sites have cert issues; acceptable for a hackathon
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} for {url}")
        return None
    except Exception as e:
        logger.error(f"Fetch failed for {url}: {e}")
        return None


def _extract_title(soup: BeautifulSoup, url: str) -> str:
    """Extract page title from HTML."""
    # Try <title> tag
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    # Try Open Graph title
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # Try <h1>
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    # Fallback to domain
    return urlparse(url).netloc


def _extract_description(soup: BeautifulSoup) -> str:
    """Extract meta description."""
    # Standard meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        return meta_desc["content"].strip()

    # Open Graph description
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        return og_desc["content"].strip()

    return ""


def _extract_links(soup: BeautifulSoup, base_url: str, max_links: int = 50) -> list[str]:
    """Extract unique links from the page."""
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        # Skip empty, anchor-only, and javascript links
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        links.add(full_url)

        if len(links) >= max_links:
            break

    return list(links)


def _fallback_extract(soup: BeautifulSoup) -> str:
    """Fallback content extraction when Trafilatura fails."""
    # Remove script, style, nav, footer, header tags
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Try to find main content area
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main:
        text = main.get_text(separator="\n", strip=True)
        # Clean up multiple newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    return ""
