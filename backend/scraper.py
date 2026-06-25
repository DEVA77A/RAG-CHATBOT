"""
WebIntel AI — Website Scraper

Strategy:
  1. BeautifulSoup — structured content extraction
  2. Combined output: title, description, clean text content, links
"""

import asyncio
import hashlib
import logging
from urllib.parse import urlparse, urljoin

import httpx
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

        # --- DOM Cleanup and Content Extraction ---
        clean_soup = BeautifulSoup(html, "lxml")
        
        # Decompose unwanted tags safely (copy list to prevent iteration mutation errors)
        unwanted_tags = ["img", "script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]
        for tag in list(clean_soup.find_all(unwanted_tags)):
            try:
                tag.decompose()
            except Exception:
                pass

        # Decompose boilerplate elements based on class/id containing keywords
        boilerplate_keywords = [
            "cookie-consent", "cookie-banner", "cookie-notice", "cookiebar",
            "consent-banner", "privacy-banner", "gdpr",
            "nav-menu", "navbar", "navigation-menu", "site-nav", "header-nav",
            "footer-nav", "footer-menu", "footer-links",
            "social-share", "share-buttons", "newsletter-signup",
            "newsletter-form", "marketing-banner", "promo-banner",
            "advertisement", "ads-container", "ad-banner"
        ]
        
        to_decompose = []
        for tag in list(clean_soup.find_all(True)):
            if tag is None:
                continue
            try:
                tag_id = (tag.get("id") or "").lower()
                if any(kw in tag_id for kw in boilerplate_keywords):
                    to_decompose.append(tag)
                    continue
                
                tag_classes = tag.get("class") or []
                if isinstance(tag_classes, list):
                    tag_classes = " ".join(tag_classes).lower()
                else:
                    tag_classes = str(tag_classes).lower()
                if any(kw in tag_classes for kw in boilerplate_keywords):
                    to_decompose.append(tag)
                    continue
            except Exception:
                pass
                
        for tag in to_decompose:
            try:
                tag.decompose()
            except Exception:
                pass
            
        # Extract text only from specified tags to preserve hierarchy
        valid_tags = ["h1", "h2", "h3", "h4", "p", "li", "td", "th"]
        
        seen = set()
        clean_lines = []
        for tag in clean_soup.find_all(valid_tags):
            name = tag.name
            line_strip = tag.get_text(separator=" ", strip=True)
            # Skip strings under 15 characters
            if len(line_strip) >= 15:
                # Filter common boilerplate phrases
                line_lower = line_strip.lower()
                if any(phrase in line_lower for phrase in [
                    "cookie policy", "we use cookies", "agree to our use of cookies",
                    "all rights reserved", "terms of service", "privacy policy",
                    "subscribe to our newsletter", "sign up for our", "skip to main content"
                ]):
                    continue
                    
                if line_strip not in seen:
                    seen.add(line_strip)
                    # Format headings to preserve hierarchy in section splits
                    if name == "h1":
                        clean_lines.append(f"# {line_strip}")
                    elif name == "h2":
                        clean_lines.append(f"## {line_strip}")
                    elif name == "h3":
                        clean_lines.append(f"### {line_strip}")
                    elif name == "h4":
                        clean_lines.append(f"#### {line_strip}")
                    elif name == "li":
                        clean_lines.append(f"* {line_strip}")
                    else:
                        clean_lines.append(line_strip)
                    
        content = "\n\n".join(clean_lines)

        if not content:
            result["error"] = "Could not extract meaningful content from this website."
            return result

        # Convert relative markdown links/images to absolute URLs
        content = _make_links_absolute(content, url)

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


def _generate_mock_pages(start_url: str) -> list[dict]:
    parsed = urlparse(start_url)
    domain = parsed.netloc or parsed.path
    if not domain:
        domain = "example.com"
        
    # Clean domain name for display
    display_name = domain.replace("www.", "").split(".")[0].title()
    
    logger.info(f"Generating fallback simulated pages for {start_url} (display name: {display_name})")
    
    pages = [
        {
            "url": start_url,
            "title": f"{display_name} — Home & Overview",
            "description": f"The main entry point for {display_name}, showcasing core services, products, and community updates.",
            "content": f"""
Welcome to {display_name}! This is the simulated homepage for {display_name} ({domain}).

{display_name} is a leading platform designed to empower developers, researchers, students, and businesses with next-generation tools, APIs, and frameworks. Our mission is to accelerate open-source innovation, artificial intelligence integration, and high-performance computing.

Core Features & Offerings:
1. Premium Developer SDK and API access for seamless backend integrations.
2. Robust student learning portals featuring courses, guides, and certification paths.
3. Open research access, containing state-of-the-art benchmarks, innovation indices, and papers.
4. Flexible subscription models and enterprise consultation plans for venture investors and companies.
5. Large community support forums and active developer discussion boards.

Get started by checking our documentation or checking out our community pages.
"""
        },
        {
            "url": urljoin(start_url, "/docs"),
            "title": f"{display_name} — Documentation & API References",
            "description": f"Technical documentation, developer guides, code tutorials, and API endpoint details for {display_name}.",
            "content": f"""
# {display_name} Technical Documentation

Welcome to the official developer references and API specifications for {display_name}.

## Quick Start Guide
To install our CLI and SDK, run:
`npm install -g @{display_name.lower()}/cli` or `pip install {display_name.lower()}-sdk`

## Authentication & API Keys
All API calls must include the authorization header:
`Authorization: Bearer <YOUR_API_KEY>`

## Endpoints:
- `POST /v1/analyze`: Analyze inputs and run evaluation metrics. Returns structural JSON output containing scores and insights.
- `GET /v1/status`: Retrieve real-time performance latency, vector db load, and active worker node status.
- `POST /v1/chat`: Connect Q&A queries to the RAG database wrapper.

## System Requirements
- Python 3.9+ or Node.js 18+
- SQLite 3.35+ or PostgreSQL 13+
- Minimum 4GB RAM for local execution.
"""
        },
        {
            "url": urljoin(start_url, "/about"),
            "title": f"{display_name} — About Us & Careers",
            "description": f"Information about the team behind {display_name}, history, mission statement, and career opportunities.",
            "content": f"""
# About {display_name}

{display_name} was founded in 2024 by a group of engineers, researchers, and educators dedicated to making advanced technologies accessible to everyone.

## Our Vision
We believe in a future where high-performance computing, generative AI, and intelligent databases are democratized and accessible via clean, standard APIs.

## Careers & Opportunities
We are actively hiring for the following roles:
- Senior Backend Developer (Python, FastAPI, SQLite, FAISS) - Remote / Hybrid.
- AI Research Scientist (LLMs, RAG, prompt engineering, evaluation metrics).
- Developer Relations Engineer - to support our growing global open-source community.
- Internships are available quarterly for computer science students.

Contact our hiring team at careers@{domain} or check our developer portal.
"""
        }
    ]
    return pages


def classify_website_type(url: str) -> str:
    """Classify site type into: 'docs', 'e-commerce', 'company', or 'general'"""
    u_lower = url.lower()
    if any(kw in u_lower for kw in ["docs", "doc", "learn", "guide", "tutorial", "wiki", "api", "reference", "tiangolo.com", "react.dev", "python.org", "github.com", "huggingface.co", "geeksforgeeks.org"]):
        return "docs"
    if any(kw in u_lower for kw in ["shop", "store", "cart", "flipkart", "amazon", "ebay", "walmart", "target", "e-commerce"]):
        return "e-commerce"
    if any(kw in u_lower for kw in ["about", "product", "features", "pricing", "pricing", "service"]):
        return "company"
    return "general"


async def crawl_website(start_url: str, max_pages: int = 10) -> list[dict]:
    """
    Recursively crawl internal pages starting from start_url in parallel batches.
    Prioritizes pages that contain keywords based on detected site type (docs, company, e-commerce).
    
    Returns:
        List of dicts, each representing a scraped page:
        [{"url": url, "title": title, "content": content, "description": desc, "content_hash": hash}, ...]
    """
    parsed_start = urlparse(start_url)
    start_domain = parsed_start.netloc
    site_type = classify_website_type(start_url)
    
    logger.info(f"Crawling start URL homepage: {start_url} (Site type: {site_type}, Max pages: {max_pages})")
    homepage_res = None
    try:
        homepage_res = await scrape_url(start_url)
    except Exception as e:
        logger.error(f"Failed to scrape start URL {start_url}: {e}")
        
    if not homepage_res or not homepage_res.get("success"):
        logger.warning(f"Could not scrape start URL homepage successfully.")
        return []
        
    pages = []
    seen_urls = set()
    seen_hashes = set()
    
    homepage_norm = start_url.split("#")[0].rstrip("/")
    seen_urls.add(homepage_norm)
    
    hp_hash = homepage_res.get("content_hash")
    if hp_hash:
        seen_hashes.add(hp_hash)
        
    pages.append({
        "url": homepage_res["url"],
        "title": homepage_res["title"],
        "description": homepage_res["description"],
        "content": homepage_res["content"],
        "content_hash": hp_hash
    })
    
    def is_ignored(url_str: str) -> bool:
        parsed = urlparse(url_str)
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        # Explicit ignore list from requirements
        ignore_kws = [
            "login", "signin", "signup", "register", "logout",
            "account", "profile", "privacy", "terms", "tos", "policy",
            "careers", "jobs", "job", "career", "cookie", "cookies",
            "terms-of-service", "privacy-policy", "sign-in", "sign-up",
            "login-page", "checkout", "cart", "terms-and-conditions"
        ]
        if any(kw in path or kw in query for kw in ignore_kws):
            return True
            
        # Social media domains
        social_domains = [
            "facebook.com", "twitter.com", "x.com", "linkedin.com",
            "instagram.com", "youtube.com", "pinterest.com", "reddit.com"
        ]
        netloc = parsed.netloc.lower()
        if any(domain in netloc for domain in social_domains) and start_domain not in netloc:
            return True
            
        # Static files
        if path.endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar.gz", ".dmg", ".exe", ".css", ".js")):
            return True
            
        return False
        
    def get_priority(url_str: str, s_type: str) -> int:
        parsed = urlparse(url_str)
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        if s_type == "docs":
            doc_kws = ["learn", "docs", "guide", "tutorial", "reference", "api", "hooks", "state", "props", "components", "context"]
            if any(x in path or x in query for x in doc_kws):
                return 80
        elif s_type == "company":
            company_kws = ["about", "products", "services", "pricing", "features", "blog", "faq", "contact"]
            if any(x in path or x in query for x in company_kws):
                return 80
        elif s_type == "e-commerce":
            ecom_kws = ["category", "product", "help", "support"]
            if any(x in path or x in query for x in ecom_kws):
                return 80
                
        # Default priority fallback (historical)
        if any(x in path or x in query for x in ["docs", "doc", "readme"]):
            return 50
        if any(x in path or x in query for x in ["api", "apis"]):
            return 45
        if "about" in path or "about" in query:
            return 40
        return 0

    # We maintain candidates as a dict: norm_url -> (full_url, priority)
    candidates = {}
    
    # Extract links from homepage to seed candidates
    for link in homepage_res.get("links", []):
        link_parsed = urlparse(link)
        is_internal = (link_parsed.netloc == start_domain) or (not link_parsed.netloc and link.startswith("/"))
        if not is_internal:
            continue
            
        full_link = link if link_parsed.netloc else urljoin(start_url, link)
        norm_link = full_link.split("#")[0].rstrip("/")
        
        if norm_link in seen_urls:
            continue
            
        if is_ignored(full_link):
            continue
            
        priority = get_priority(full_link, site_type)
        candidates[norm_link] = (full_link, priority)
        
    # BFS multi-level parallelized crawling loop
    while len(pages) < max_pages and candidates:
        # Sort candidates by priority descending
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][1], reverse=True)
        
        # Take a batch of up to min(10, max_pages - len(pages)) candidates
        batch_size = min(10, max_pages - len(pages))
        batch = sorted_candidates[:batch_size]
        
        # Remove batch from candidates and mark as seen
        for norm_link, _ in batch:
            del candidates[norm_link]
            seen_urls.add(norm_link)
            
        # Concurrently scrape the batch
        logger.info(f"Concurrently scraping {len(batch)} prioritized internal pages.")
        tasks = [scrape_url(info[0]) for norm_link, info in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, Exception) or not res or not res.get("success"):
                continue
                
            c_hash = res.get("content_hash")
            if c_hash in seen_hashes:
                logger.info(f"Skipping duplicate content page: {res['url']}")
                continue
                
            seen_hashes.add(c_hash)
            pages.append({
                "url": res["url"],
                "title": res["title"],
                "description": res["description"],
                "content": res["content"],
                "content_hash": c_hash
            })
            
            # Extract links from the newly crawled page to discover deeper pages
            if len(pages) < max_pages:
                for link in res.get("links", []):
                    link_parsed = urlparse(link)
                    is_internal = (link_parsed.netloc == start_domain) or (not link_parsed.netloc and link.startswith("/"))
                    if not is_internal:
                        continue
                    full_link = link if link_parsed.netloc else urljoin(res["url"], link)
                    norm_link = full_link.split("#")[0].rstrip("/")
                    
                    if norm_link in seen_urls or norm_link in candidates or is_ignored(full_link):
                        continue
                        
                    priority = get_priority(full_link, site_type)
                    candidates[norm_link] = (full_link, priority)
                    
    return pages


def _make_links_absolute(text: str, base_url: str) -> str:
    """Convert relative markdown links and images to absolute URLs relative to base_url."""
    import re
    from urllib.parse import urljoin
    
    if not text:
        return ""
        
    # Match optional ! followed by [link text](url)
    pattern = r'(!?)\[([^\]]*)\]\(([^)]*)\)'
    
    def replacer(match):
        img_prefix = match.group(1)
        link_text = match.group(2)
        link_url = match.group(3).strip()
        
        # If it starts with scheme, tel, mailto or is fragment, leave it
        if (link_url.startswith(("http://", "https://", "mailto:", "tel:", "#")) 
            or not link_url):
            return match.group(0)
            
        abs_url = urljoin(base_url, link_url)
        return f"{img_prefix}[{link_text}]({abs_url})"
        
    return re.sub(pattern, replacer, text)
