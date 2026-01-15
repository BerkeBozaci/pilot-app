#!/usr/bin/env python3
"""
search_furniture.py
-------------------
Uses a self-hosted SearXNG instance to find "similar product links" from furniture descriptions.
Enriches each result by fetching the product page and extracting:
  - title
  - image (multiple fallback methods)
  - price + currency (JSON-LD, microdata, regex patterns)

Requirements:
  pip install requests beautifulsoup4 lxml

Run:
  python3 search_furniture.py --in furniture.json --out results.json \
    --searx http://localhost:8080 \
    --sites ikea.com,wayfair.co.uk,johnlewis.com \
    --max 6 --lang en --enrich --fetch-top 4

Key improvements:
  - Multiple image extraction methods (JSON-LD, OG, meta, img tags)
  - Multiple price extraction methods (JSON-LD, microdata, regex)
  - Better query building for furniture
  - Fallback queries when no results
  - Currency detection for multiple regions
"""

from __future__ import annotations

import argparse
import json
import re
import time
from urllib.parse import urlparse, urljoin
from typing import Optional

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Constants
# ----------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# Price patterns for various currencies
PRICE_PATTERNS = [
    # Turkish Lira
    r"([\d.,]+)\s*(?:TL|₺|TRY)",
    r"(?:TL|₺|TRY)\s*([\d.,]+)",
    # Euro
    r"€\s*([\d.,]+)",
    r"([\d.,]+)\s*€",
    r"EUR\s*([\d.,]+)",
    # British Pound
    r"£\s*([\d.,]+)",
    r"([\d.,]+)\s*£",
    r"GBP\s*([\d.,]+)",
    # US Dollar
    r"\$\s*([\d.,]+)",
    r"([\d.,]+)\s*\$",
    r"USD\s*([\d.,]+)",
    # Generic price-like patterns
    r'"price":\s*"?([\d.,]+)"?',
    r'"amount":\s*"?([\d.,]+)"?',
]

CURRENCY_SYMBOLS = {
    "₺": "TRY",
    "TL": "TRY",
    "TRY": "TRY",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "$": "USD",
    "USD": "USD",
}


# ----------------------------
# Helpers
# ----------------------------


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def blocked(domain: str, blocklist: list[str]) -> bool:
    d = (domain or "").lower()
    for b in blocklist:
        b = b.strip().lower()
        if not b:
            continue
        if d == b or d.endswith("." + b):
            return True
    return False


def is_probably_product_page(url: str) -> bool:
    """Heuristic: prefer product pages over category pages."""
    u = (url or "").lower()

    # Good signs - likely a product page
    good_signs = [
        "/p/",
        "/p-",
        "/product/",
        "/products/",
        "/item/",
        "/dp/",
        "/urun/",
        "-p-",
        "/pd/",
    ]
    if any(x in u for x in good_signs):
        return True

    # Bad signs - likely a category page
    bad_signs = [
        "/search",
        "/s?",
        "k=",
        "/cat/",
        "/category",
        "/categories",
        "/collections",
        "/c/",
        "/katalog",
        "/liste",
        "/list",
        "/query",
        "/browse",
        "/shop/",
        "?q=",
        "&q=",
    ]
    if any(x in u for x in bad_signs):
        return False

    return True  # Default: might be product


def clean_price(price_str: str) -> Optional[str]:
    """Clean price string to numeric format."""
    if not price_str:
        return None
    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[^\d.,]", "", str(price_str))
    if not cleaned:
        return None
    # Handle European format (1.234,56) vs US format (1,234.56)
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            # European format: 1.234,56
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            # US format: 1,234.56
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # Could be European decimal or US thousands separator
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            # Likely European decimal: 234,56
            cleaned = cleaned.replace(",", ".")
        else:
            # Likely US thousands: 1,234
            cleaned = cleaned.replace(",", "")
    return cleaned if cleaned else None


def detect_currency(text: str, url: str = "") -> str:
    """Detect currency from text or URL."""
    text = (text or "").upper()
    url = (url or "").lower()

    # Check URL domain for hints
    if ".tr" in url or "trendyol" in url or "hepsiburada" in url:
        return "TRY"
    if ".co.uk" in url or "johnlewis" in url or "dunelm" in url:
        return "GBP"
    if ".de" in url or "home24" in url:
        return "EUR"

    # Check text for currency symbols
    for symbol, currency in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return currency

    return "USD"  # Default


# ----------------------------
# Query Building
# ----------------------------


def build_queries(item: dict, sites: list[str]) -> list[str]:
    """Build search queries from furniture item data."""
    name = norm_text(item.get("name", ""))
    color = norm_text(item.get("color", ""))
    material = norm_text(item.get("material", ""))
    style = norm_text(item.get("style", ""))

    # Extract useful material tokens
    material_tokens = []
    m_low = material.lower()

    material_keywords = [
        ("walnut", "walnut"),
        ("oak", "oak"),
        ("pine", "pine"),
        ("wood", "wood"),
        ("metal", "metal"),
        ("leather", "leather"),
        ("fabric", "fabric"),
        ("velvet", "velvet"),
        ("jute", "jute"),
        ("glass", "glass"),
        ("marble", "marble"),
        ("rattan", "rattan"),
    ]

    for keyword, token in material_keywords:
        if keyword in m_low:
            material_tokens.append(token)

    # Special handling for "black metal frame"
    if "black metal" in m_low:
        material_tokens.append('"black metal frame"')
    elif "black" in m_low and "frame" in m_low:
        material_tokens.append("black frame")

    # Style tokens
    style_token = ""
    s_low = style.lower()
    if "mid" in s_low and "century" in s_low:
        style_token = "mid-century"
    elif "scandinav" in s_low:
        style_token = "scandinavian"
    elif "industrial" in s_low:
        style_token = "industrial"

    # Site restriction
    site_part = ""
    if sites:
        site_part = (
            "(" + " OR ".join([f"site:{s.strip()}" for s in sites if s.strip()]) + ") "
        )

    # Build multiple query variations
    mat_str = " ".join(material_tokens[:3])  # Limit material tokens

    queries = []

    # Query 1: Full details
    q1 = f"{site_part}{name} {color} {mat_str}".strip()
    if style_token:
        q1 += f" {style_token}"
    queries.append(norm_text(q1))

    # Query 2: Name + color + material (no style)
    q2 = f"{site_part}{name} {color} {mat_str}".strip()
    queries.append(norm_text(q2))

    # Query 3: Name + color only
    q3 = f"{site_part}{name} {color}".strip()
    queries.append(norm_text(q3))

    # Query 4: Name + material only
    q4 = f"{site_part}{name} {mat_str}".strip()
    queries.append(norm_text(q4))

    # Query 5: Just the name (fallback)
    q5 = f"{site_part}{name}".strip()
    queries.append(norm_text(q5))

    # Dedupe while preserving order
    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            out.append(q)
            seen.add(q)

    return out


# ----------------------------
# SearXNG Search
# ----------------------------


def searx_search(
    searx_base: str,
    query: str,
    max_results: int = 10,
    lang: str = "en",
    timeout: int = 30,
) -> list[dict]:
    """Search using SearXNG instance."""
    url = searx_base.rstrip("/") + "/search"
    params = {
        "q": query,
        "format": "json",
        "language": lang,
        "safesearch": 0,
    }

    try:
        r = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        print(f"    [WARN] SearXNG timeout for query: {query[:50]}...")
        return []
    except Exception as e:
        print(f"    [ERROR] SearXNG error: {e}")
        return []

    results = []
    for item in data.get("results", [])[:max_results]:
        u = item.get("url")
        if not u:
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": u,
                "snippet": item.get("content") or item.get("snippet"),
                "domain": domain_of(u),
                "thumbnail": item.get("thumbnail"),  # SearXNG sometimes provides this
            }
        )

    return results


# ----------------------------
# Enrichment - Image Extraction
# ----------------------------


def extract_images_from_jsonld(jsonlds: list[dict], base_url: str) -> list[str]:
    """Extract images from JSON-LD data."""
    images = []

    for data in jsonlds:
        # Handle @graph structure
        if "@graph" in data and isinstance(data["@graph"], list):
            for item in data["@graph"]:
                if isinstance(item, dict):
                    images.extend(_get_images_from_node(item, base_url))
        images.extend(_get_images_from_node(data, base_url))

    return images


def _get_images_from_node(node: dict, base_url: str) -> list[str]:
    """Extract images from a single JSON-LD node."""
    images = []

    # Direct image field
    img = node.get("image")
    if img:
        if isinstance(img, str):
            images.append(_normalize_url(img, base_url))
        elif isinstance(img, list):
            for i in img:
                if isinstance(i, str):
                    images.append(_normalize_url(i, base_url))
                elif isinstance(i, dict) and i.get("url"):
                    images.append(_normalize_url(i["url"], base_url))
        elif isinstance(img, dict) and img.get("url"):
            images.append(_normalize_url(img["url"], base_url))

    # Thumbnail
    thumb = node.get("thumbnail")
    if isinstance(thumb, str):
        images.append(_normalize_url(thumb, base_url))
    elif isinstance(thumb, dict) and thumb.get("url"):
        images.append(_normalize_url(thumb["url"], base_url))

    return [i for i in images if i]


def extract_images_from_meta(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract images from meta tags."""
    images = []

    # OG and Twitter images
    meta_props = ["og:image", "og:image:url", "og:image:secure_url"]
    meta_names = ["twitter:image", "twitter:image:src"]

    for prop in meta_props:
        tag = soup.find("meta", property=prop)
        if tag and tag.get("content"):
            images.append(_normalize_url(tag["content"], base_url))

    for name in meta_names:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            images.append(_normalize_url(tag["content"], base_url))

    return [i for i in images if i]


def extract_images_from_html(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract product images from HTML img tags."""
    images = []

    # Look for product image containers (common patterns)
    selectors = [
        # Common product image selectors
        'img[class*="product"]',
        'img[class*="gallery"]',
        'img[class*="main"]',
        'img[id*="product"]',
        "img[data-src]",
        # Container-based
        '[class*="product-image"] img',
        '[class*="gallery"] img',
        '[class*="main-image"] img',
        '[id*="product-image"] img',
        # IKEA specific
        'img[class*="pip-image"]',
        # Wayfair specific
        'img[class*="ProductDetailImage"]',
    ]

    for selector in selectors:
        try:
            for img in soup.select(selector)[:5]:  # Limit to 5 per selector
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if src and not _is_tiny_image(src):
                    images.append(_normalize_url(src, base_url))
        except Exception:
            continue

    # Fallback: get largest images
    if not images:
        for img in soup.find_all("img")[:20]:
            src = img.get("src") or img.get("data-src")
            if src and not _is_tiny_image(src):
                # Check for size hints
                width = img.get("width", "0")
                height = img.get("height", "0")
                try:
                    if int(width) >= 200 or int(height) >= 200:
                        images.append(_normalize_url(src, base_url))
                except (ValueError, TypeError):
                    # No size info, check URL patterns
                    if any(
                        x in src.lower()
                        for x in ["large", "big", "main", "product", "gallery"]
                    ):
                        images.append(_normalize_url(src, base_url))

    return images


def _normalize_url(url: str, base_url: str) -> str:
    """Normalize relative URLs to absolute."""
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urljoin(base_url, url)
    if not url.startswith("http"):
        return urljoin(base_url, url)
    return url


def _is_tiny_image(url: str) -> bool:
    """Check if URL suggests a tiny/icon image."""
    url_lower = url.lower()
    skip_patterns = [
        "icon",
        "logo",
        "sprite",
        "pixel",
        "tracking",
        "blank",
        "1x1",
        "spacer",
        "placeholder",
        "avatar",
        "profile",
        ".svg",
        "data:image",
        "base64",
    ]
    return any(p in url_lower for p in skip_patterns)


# ----------------------------
# Enrichment - Price Extraction
# ----------------------------


def extract_price_from_jsonld(
    jsonlds: list[dict],
) -> tuple[Optional[str], Optional[str]]:
    """Extract price from JSON-LD Product schema."""
    for data in jsonlds:
        # Handle @graph
        if "@graph" in data and isinstance(data["@graph"], list):
            for item in data["@graph"]:
                price, currency = _get_price_from_node(item)
                if price:
                    return price, currency

        price, currency = _get_price_from_node(data)
        if price:
            return price, currency

    return None, None


def _get_price_from_node(node: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract price from a JSON-LD node."""
    if not isinstance(node, dict):
        return None, None

    # Check if this is a Product type
    node_type = node.get("@type", "")
    if isinstance(node_type, list):
        node_type = " ".join(str(t) for t in node_type)

    offers = node.get("offers")
    if not offers:
        return None, None

    # Offers can be a single object or a list
    if isinstance(offers, list):
        offers = offers[0] if offers else {}

    if isinstance(offers, dict):
        price = offers.get("price") or offers.get("lowPrice") or offers.get("highPrice")
        currency = offers.get("priceCurrency")

        if price is not None:
            return clean_price(str(price)), currency

    return None, None


def extract_price_from_microdata(
    soup: BeautifulSoup,
) -> tuple[Optional[str], Optional[str]]:
    """Extract price from microdata (itemprop)."""
    # Try various price microdata patterns
    price_selectors = [
        '[itemprop="price"]',
        '[itemprop="lowPrice"]',
        '[class*="price"]',
        "[data-price]",
        '[id*="price"]',
    ]

    for selector in price_selectors:
        try:
            for elem in soup.select(selector)[:3]:
                price_text = (
                    elem.get("content") or elem.get("data-price") or elem.get_text()
                )
                price = clean_price(price_text)
                if price:
                    # Try to find currency nearby
                    currency = elem.get("data-currency")
                    if not currency:
                        currency_elem = soup.select_one('[itemprop="priceCurrency"]')
                        if currency_elem:
                            currency = (
                                currency_elem.get("content") or currency_elem.get_text()
                            )
                    return price, currency.strip() if currency else None
        except Exception:
            continue

    return None, None


def extract_price_from_text(html: str, url: str) -> tuple[Optional[str], Optional[str]]:
    """Extract price using regex patterns."""
    for pattern in PRICE_PATTERNS:
        matches = re.findall(pattern, html, re.IGNORECASE)
        if matches:
            price = clean_price(matches[0])
            if price:
                try:
                    # Validate it's a reasonable price (not a year, etc.)
                    price_float = float(price)
                    if 1 < price_float < 1000000:
                        currency = detect_currency(html[:1000], url)
                        return price, currency
                except ValueError:
                    continue

    return None, None


# ----------------------------
# Main Enrichment Function
# ----------------------------


def parse_jsonld(soup: BeautifulSoup) -> list[dict]:
    """Parse all JSON-LD blocks from page."""
    results = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            raw = tag.string or tag.get_text() or ""
            raw = raw.strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, list):
                results.extend([d for d in data if isinstance(d, dict)])
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            continue
    return results


def enrich_url(session: requests.Session, url: str, cache: dict) -> dict:
    """Fetch URL and extract title, image, price."""
    if url in cache:
        return cache[url]

    result = {
        "final_url": url,
        "title": None,
        "image": None,
        "images": [],
        "price": None,
        "currency": None,
        "success": False,
    }

    try:
        r = session.get(url, headers=DEFAULT_HEADERS, timeout=25, allow_redirects=True)
        result["final_url"] = r.url

        if r.status_code >= 400:
            cache[url] = result
            return result

        html = r.text or ""
        if len(html) < 500:  # Too small, probably blocked
            cache[url] = result
            return result

        soup = BeautifulSoup(html, "lxml")
        base_url = result["final_url"]

        # Parse JSON-LD
        jsonlds = parse_jsonld(soup)

        # --- Extract Title ---
        # 1. From meta tags
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            result["title"] = og_title["content"].strip()

        # 2. From JSON-LD Product name
        if not result["title"]:
            for jd in jsonlds:
                if isinstance(jd.get("name"), str):
                    result["title"] = jd["name"]
                    break
                if "@graph" in jd:
                    for item in jd.get("@graph", []):
                        if isinstance(item, dict) and isinstance(item.get("name"), str):
                            result["title"] = item["name"]
                            break

        # 3. From <title> tag
        if not result["title"]:
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text().strip()

        # --- Extract Images ---
        all_images = []

        # 1. From JSON-LD
        all_images.extend(extract_images_from_jsonld(jsonlds, base_url))

        # 2. From meta tags
        all_images.extend(extract_images_from_meta(soup, base_url))

        # 3. From HTML
        all_images.extend(extract_images_from_html(soup, base_url))

        # Dedupe and filter
        seen_images = set()
        unique_images = []
        for img in all_images:
            if img and img not in seen_images and not _is_tiny_image(img):
                seen_images.add(img)
                unique_images.append(img)

        result["images"] = unique_images[:5]  # Keep top 5
        result["image"] = unique_images[0] if unique_images else None

        # --- Extract Price ---
        # 1. From JSON-LD
        price, currency = extract_price_from_jsonld(jsonlds)

        # 2. From microdata
        if not price:
            price, currency = extract_price_from_microdata(soup)

        # 3. From text patterns (last resort)
        if not price:
            price, currency = extract_price_from_text(html, base_url)

        result["price"] = price
        result["currency"] = currency or detect_currency(html[:2000], base_url)

        result["success"] = True

    except requests.exceptions.Timeout:
        print(f"      [TIMEOUT] {url[:60]}...")
    except Exception as e:
        print(f"      [ERROR] {url[:40]}... - {str(e)[:50]}")

    cache[url] = result
    return result


# ----------------------------
# Main
# ----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Furniture search with image/price extraction"
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input furniture JSON")
    ap.add_argument("--out", dest="outp", default="results.json", help="Output JSON")
    ap.add_argument("--searx", default="http://localhost:8080", help="SearXNG base URL")
    ap.add_argument("--sites", default="", help="Comma-separated allowed domains")
    ap.add_argument(
        "--block",
        default="pinterest.com,facebook.com,instagram.com,twitter.com",
        help="Comma-separated blocked domains",
    )
    ap.add_argument(
        "--max", dest="maxr", type=int, default=6, help="Max results per item"
    )
    ap.add_argument("--lang", default="en", help="Search language (en/tr etc.)")
    ap.add_argument(
        "--enrich", action="store_true", help="Fetch and extract image/price"
    )
    ap.add_argument(
        "--fetch-top",
        dest="fetch_top",
        type=int,
        default=4,
        help="How many top results to enrich per item",
    )
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between fetches")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = ap.parse_args()

    # Load input
    with open(args.inp, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    raw_items = input_data.get("items") if isinstance(input_data, dict) else input_data
    if not isinstance(raw_items, list):
        raise SystemExit(
            "Input JSON must contain a list under key 'items' or be a list itself."
        )

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    blocklist = [s.strip() for s in args.block.split(",") if s.strip()]

    print(f"[*] SearXNG: {args.searx}")
    print(f"[*] Sites filter: {sites if sites else 'none (searching all)'}")
    print(f"[*] Enrichment: {'enabled' if args.enrich else 'disabled'}")
    print(f"[*] Processing {len(raw_items)} items...")
    print()

    output = {
        "source": "searxng",
        "items": [],
        "meta": {
            "searx": args.searx,
            "sites": sites,
            "blocked": blocklist,
            "enrich": args.enrich,
            "fetch_top": args.fetch_top,
            "language": args.lang,
        },
    }

    session = requests.Session()
    fetch_cache: dict[str, dict] = {}

    total_results = 0
    total_enriched = 0

    for idx, item in enumerate(raw_items, 1):
        item_name = item.get("name", "unknown")
        print(f"[{idx}/{len(raw_items)}] {item_name}")

        queries = build_queries(item, sites)
        gathered: list[dict] = []

        for q_idx, q in enumerate(queries):
            if args.verbose:
                print(f"  Query {q_idx+1}: {q[:80]}...")

            try:
                results = searx_search(
                    args.searx, q, max_results=args.maxr * 2, lang=args.lang
                )
            except Exception as e:
                print(f"    [ERROR] {e}")
                results = []

            # Filter and process results
            for r in results:
                d = r.get("domain", "")
                if blocked(d, blocklist):
                    continue
                r["query_used"] = q
                gathered.append(r)

            # Dedupe by URL
            seen_urls = set()
            unique = []
            for r in gathered:
                u = r.get("url")
                if u and u not in seen_urls:
                    unique.append(r)
                    seen_urls.add(u)
            gathered = unique

            if len(gathered) >= args.maxr:
                break

            time.sleep(0.3)  # Small delay between queries

        # Sort: prefer product pages
        gathered.sort(
            key=lambda x: 0 if is_probably_product_page(x.get("url", "")) else 1
        )
        gathered = gathered[: args.maxr]

        print(f"  Found: {len(gathered)} results")
        total_results += len(gathered)

        # Enrich top N results
        if args.enrich and gathered:
            enrich_count = min(args.fetch_top, len(gathered))
            print(f"  Enriching top {enrich_count}...")

            for i in range(enrich_count):
                url = gathered[i].get("url")
                if not url:
                    continue

                if args.verbose:
                    print(f"    [{i+1}] {url[:60]}...")

                enriched = enrich_url(session, url, fetch_cache)
                gathered[i]["enriched"] = enriched

                if enriched.get("success"):
                    total_enriched += 1
                    if args.verbose:
                        img = (
                            enriched.get("image", "")[:50]
                            if enriched.get("image")
                            else "none"
                        )
                        price = enriched.get("price", "none")
                        print(f"        ✓ Image: {img}... | Price: {price}")

                time.sleep(args.sleep)

        output["items"].append(
            {
                "input_item": item,
                "queries": queries,
                "results": gathered,
            }
        )

        print()

    # Save output
    with open(args.outp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"[OK] Saved: {args.outp}")
    print(f"[*] Total items: {len(raw_items)}")
    print(f"[*] Total results: {total_results}")
    if args.enrich:
        print(f"[*] Successfully enriched: {total_enriched}")


if __name__ == "__main__":
    main()
