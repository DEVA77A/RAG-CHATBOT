"""
WebIntel AI — Website Scraper

Strategy:
  1. Trafilatura (primary) — excellent at extracting main article content
  2. BeautifulSoup (fallback) — raw HTML parsing for metadata and links
  3. Combined output: title, description, clean text content, links
"""

import asyncio
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
            timeout=7.0,
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


async def crawl_website(start_url: str, max_pages: int = 6) -> list[dict]:
    """
    Recursively crawl internal pages starting from start_url in parallel batches.
    Prioritizes pages that contain keywords like 'about', 'product', 'docs', etc.
    
    Returns:
        List of dicts, each representing a scraped page:
        [{"url": url, "title": title, "content": content, "description": desc}, ...]
    """
    parsed_start = urlparse(start_url)
    start_domain = parsed_start.netloc
    
    visited = set()
    pages = []
    
    # 1. Fetch homepage first
    norm_start = start_url.split("#")[0].rstrip("/")
    visited.add(norm_start)
    
    logger.info(f"Crawling start URL homepage: {start_url}")
    try:
        homepage_res = await scrape_url(start_url)
    except Exception as e:
        logger.error(f"Failed to scrape start URL {start_url}: {e}")
        return []
        
    if not homepage_res or not homepage_res.get("success"):
        logger.warning(f"Could not scrape start URL homepage successfully")
        return []
        
    pages.append({
        "url": homepage_res["url"],
        "title": homepage_res["title"],
        "description": homepage_res["description"],
        "content": homepage_res["content"]
    })
    
    # Queue stores (url, priority)
    queue = []
    
    def add_links_to_queue(links, current_url):
        for link in links:
            link_parsed = urlparse(link)
            # Ensure it's internal (same domain)
            if link_parsed.netloc == start_domain or (not link_parsed.netloc and link.startswith("/")):
                # Resolve relative link if necessary
                full_link = link if link_parsed.netloc else urljoin(current_url, link)
                full_link_parsed = urlparse(full_link)
                link_norm = full_link.split("#")[0].rstrip("/")
                
                if link_norm not in visited:
                    # Determine priority
                    path = full_link_parsed.path.lower()
                    query = full_link_parsed.query.lower()
                    priority = 0
                    
                    # Keywords to prioritize
                    keywords = [
                        "about", "product", "docs", "doc", "api", "pricing", 
                        "features", "faq", "guide", "learn", "download", 
                        "readme", "developer", "support"
                    ]
                    if any(kw in path or kw in query for kw in keywords):
                        priority = 1
                    
                    # Check if already in queue, keep highest priority
                    in_queue_idx = -1
                    for idx, (q_url, q_pri) in enumerate(queue):
                        if q_url.split("#")[0].rstrip("/") == link_norm:
                            in_queue_idx = idx
                            break
                    if in_queue_idx != -1:
                        if priority > queue[in_queue_idx][1]:
                            queue[in_queue_idx] = (full_link, priority)
                    else:
                        queue.append((full_link, priority))

    add_links_to_queue(homepage_res.get("links", []), start_url)
    
    # Process queue in parallel batches
    batch_size = 5
    
    while queue and len(pages) < max_pages:
        # Sort queue by priority descending
        queue.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many pages we still need
        needed = max_pages - len(pages)
        current_batch_size = min(batch_size, needed, len(queue))
        
        batch = []
        for _ in range(current_batch_size):
            batch.append(queue.pop(0))
            
        # Extract URLs and filter out if already visited (in case of duplicates)
        batch_urls = []
        for u, pri in batch:
            norm = u.split("#")[0].rstrip("/")
            if norm not in visited:
                visited.add(norm)
                batch_urls.append(u)
                
        if not batch_urls:
            continue
            
        logger.info(f"Crawling batch of {len(batch_urls)} URLs in parallel: {batch_urls}")
        
        # Fetch batch in parallel
        tasks = [scrape_url(u) for u in batch_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error in batch scrape task: {res}")
                continue
            if not res or not res.get("success"):
                continue
                
            pages.append({
                "url": res["url"],
                "title": res["title"],
                "description": res["description"],
                "content": res["content"]
            })
            
            # Extract links and add to queue
            add_links_to_queue(res.get("links", []), res["url"])
            
    return pages
