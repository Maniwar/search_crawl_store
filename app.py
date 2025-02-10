import nest_asyncio
nest_asyncio.apply()
import os
import subprocess
import sys

# Install Playwright without requiring sudo.
def install_playwright():
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True, capture_output=True)

install_playwright()

import asyncio
import json
import requests
import re
import streamlit as st
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin, urlunparse
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI

# Advanced imports for Crawl4AI, dispatchers, and rate limiting
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    RateLimiter,
    CrawlerMonitor,
    DisplayMode
)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from collections import deque

load_dotenv()

# Environment variables and client setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in your environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

#############################
# Install JS snippet (if needed)
#############################
js_click_all = """
(async () => {
    const clickable = document.querySelectorAll("a, button");
    for (let el of clickable) {
        try {
            el.click();
            await new Promise(r => setTimeout(r, 300));
        } catch(e) {}
    }
})();
"""

#############################
# Helper Functions
#############################

def normalize_url(u: str) -> str:
    parts = urlparse(u)
    normalized_path = parts.path.rstrip('/') if parts.path != '/' else parts.path
    normalized = parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower(), path=normalized_path)
    return urlunparse(normalized)

def chunk_text(t: str, max_chars: int = 5000) -> List[str]:
    paragraphs = t.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += ("\n\n" if current_chunk else "") + para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

#############################
# OpenAI Embedding
#############################
async def get_embedding(text: str) -> Optional[List[float]]:
    try:
        r = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return r.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def extract_reference_snippet(content: str, query: str, snippet_length: int = 300) -> str:
    query_terms = query.split()
    first_index = None
    for term in query_terms:
        match = re.search(re.escape(term), content, flags=re.IGNORECASE)
        if match:
            if first_index is None or match.start() < first_index:
                first_index = match.start()
    if first_index is not None:
        start = max(0, first_index - snippet_length // 2)
        snippet = content[start:start + snippet_length]
    else:
        snippet = content[:snippet_length]

    def highlight_word(word):
        if word.lower().startswith("http"):
            return word
        for term in query_terms:
            if re.search(re.escape(term), word, flags=re.IGNORECASE):
                return f'<span style="background-color: yellow;">{word}</span>'
        return word

    highlighted = " ".join(highlight_word(w) for w in snippet.split())
    return highlighted

#############################
# Retrieve multiple matches
#############################
def retrieve_relevant_documents(query: str, n_matches: int = 5, max_snippet_len: int = 500) -> str:
    e = asyncio.run(get_embedding(query))
    if e is None:
        return "Error generating embedding for the query."

    r = supabase.rpc("match_documents", {
        "query_embedding": e,
        "match_count": n_matches
    }).execute()

    docs = r.data
    if not docs:
        return "No relevant documentation found."

    combined_snippets = []
    for doc in docs:
        content_slice = doc["content"][:max_snippet_len]
        snippet = extract_reference_snippet(content_slice, query)
        snippet_block = f"""\n# {doc['title']}\n\n{snippet}\n...\nSource: {doc['url']}\nSimilarity: {doc['similarity']:.3f}\n""".strip()
        combined_snippets.append(snippet_block)

    return "\n".join(combined_snippets)

#############################
# Sitemap Helpers
#############################

def get_urls_from_sitemap(u: str) -> List[str]:
    try:
        r = requests.get(u, stream=True, timeout=15)
        r.raise_for_status()

        content_type = r.headers.get('Content-Type', '').lower()
        if 'xml' not in content_type and 'text/xml' not in content_type:
            st.warning(f"Warning: URL {u} does not appear to be an XML sitemap (Content-Type: {content_type}). Skipping XML parsing.")
            return []

        xml_content = r.content
        try:
            ro = ElementTree.fromstring(xml_content)
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            return [loc.text.strip() for loc in ro.findall(".//ns:loc", ns)]
        except ElementTree.ParseError as parse_err:
            st.error(f"Sitemap XML Parse Error for URL: {u}")
            st.error(f"ParseError details: {parse_err}")
            print("Problematic XML Content (truncated):\n", xml_content[:500].decode('utf-8', errors='ignore'))
            return []
    except requests.exceptions.RequestException as req_err:
        st.error(f"Sitemap Request Error for URL: {u}: {req_err}")
        return []
    except Exception as e:
        st.error(f"Sitemap General Error for URL: {u}: {e}")
        return []

def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc

def format_sitemap_url(u: str) -> str:
    if "sitemap.xml" in u:
        return u
    parsed = urlparse(u)
    if parsed.path in ("", "/"):
        return f"{u.rstrip('/')}" + "/sitemap.xml"
    return u

#############################
# Best practice config from docs
#############################
def get_run_config(with_js: bool = False, delay: float = 1.0, word_threshold: int = 50) -> CrawlerRunConfig:
    kwargs = {
        "cache_mode": CacheMode.BYPASS,
        "stream": False,
        "exclude_external_links": False,
        "wait_for_images": True,
        "delay_before_return_html": delay,
        "excluded_tags": ["header", "footer", "nav", "aside"],
        "word_count_threshold": word_threshold,
    }
    if with_js:
        kwargs["js_code"] = [js_click_all]
    return CrawlerRunConfig(**kwargs)

def extract_title_and_summary_from_markdown(md: str) -> Dict[str, str]:
    lines = md.splitlines()
    title = "Untitled"
    for line in lines:
        if line.strip().startswith("#"):
            title = line.lstrip("# ").strip()
            break
    return {"title": title, "summary": md}

#############################
# Processing & Storing docs
#############################
async def process_chunk(chunk: str, num: int, url: str) -> Dict[str, Any]:
    ts = extract_title_and_summary_from_markdown(chunk)
    embedding = await get_embedding(ts["summary"])
    if embedding is None:
        st.warning(f"Skipping chunk {num} from {url} due to embedding error.")
        return None

    return {
        "id": f"{url}_{num}",
        "url": url,
        "chunk_number": num,
        "title": ts["title"],
        "summary": ts["summary"],
        "content": chunk,
        "metadata": {
            "source": urlparse(url).netloc,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
        },
        "embedding": embedding,
    }

async def insert_chunk_to_supabase(d: Dict[str, Any]):
    if d is None:
        return
    try:
        supabase.table("rag_chunks").upsert(d).execute()
    except Exception as e:
        st.error(f"Insert error to Supabase: {e}")

async def process_and_store_document(url: str, md: str):
    chunks = chunk_text(md, max_chars=st.session_state.chunk_max_chars)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk_to_supabase(item) for item in processed_chunks if item is not None]
    await asyncio.gather(*insert_tasks)

#############################
# BFS Link Discovery (Optional)
#############################
async def discover_internal_links(start_urls: List[str], max_depth: int = 3) -> List[str]:
    visited = set()
    discovered = set()
    queue = deque((url, 0) for url in start_urls)

    bc = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    run_conf = get_run_config(with_js=st.session_state.use_js_for_crawl, delay=st.session_state.crawl_delay, word_threshold=st.session_state.crawl_word_threshold)

    async with AsyncWebCrawler(config=bc) as crawler:
        progress_bar = st.progress(0.0)
        total_urls = len(start_urls)
        urls_processed_count = 0

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            result = await crawler.arun(url=url, config=run_conf)
            urls_processed_count += 1
            progress_bar.progress(min(1.0, urls_processed_count / total_urls) if total_urls > 0 else 0.0)

            if result.success:
                discovered.add(url)
                links_dict = getattr(result, "links", {})
                internal_links = links_dict.get("internal", [])
                for link_obj in internal_links:
                    href = link_obj.get("href")
                    if href:
                        abs_url = urljoin(url, href)
                        if abs_url and same_domain(abs_url, url):
                            queue.append((abs_url, depth + 1))
            else:
                st.error(f"Link discovery failed for URL: {url}. Error: {result.error_message}")

        progress_bar.empty()
        return list(discovered)

#############################
# Parallel Crawl (arun_many)
#############################
async def crawl_parallel(urls: List[str], max_concurrent: int = 10):
    if not urls:
        st.warning("Warning: crawl_parallel received an empty URL list. Nothing to crawl.")
        return

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
        rate_limiter=RateLimiter(
            base_delay=(st.session_state.rate_limiter_base_delay_min, st.session_state.rate_limiter_base_delay_max),
            max_delay=st.session_state.rate_limiter_max_delay,
            max_retries=st.session_state.rate_limiter_max_retries,
            rate_limit_codes=[429, 503]
        ),
        monitor=CrawlerMonitor(
            max_visible_rows=15,
            display_mode=DisplayMode.DETAILED
        )
    )
    bc = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    run_conf = get_run_config(with_js=st.session_state.use_js_for_crawl, delay=st.session_state.crawl_delay, word_threshold=st.session_state.crawl_word_threshold)

    progress_bar = st.progress(0.0)
    total_urls = len(urls)
    urls_crawled_count = 0

    async with AsyncWebCrawler(config=bc) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=run_conf,
            dispatcher=dispatcher
        )
        for r in results:
            urls_crawled_count += 1
            progress_bar.progress(min(1.0, urls_crawled_count / total_urls) if total_urls > 0 else 0.0)
            if r.success:
                md = r.markdown_v2.raw_markdown
                await process_and_store_document(r.url, md)
            else:
                st.error(f"Failed crawling {r.url}: {r.error_message}")
    progress_bar.empty()

#############################
# DB & Stats
#############################
def delete_all_chunks():
    if st.warning("Are you sure you want to clear the database? This action cannot be undone.", type="warning"):
        supabase.table("rag_chunks").delete().neq("id", "").execute()
        st.session_state.processing_complete = False
        st.session_state.urls_processed = set()
        st.session_state.messages = []
        st.session_state.suggested_questions = None
        st.success("Database cleared successfully!")
        st.rerun()

def get_db_stats():
    try:
        r = supabase.table("rag_chunks").select("id, url, metadata").execute()
        d = r.data
        if not d:
            return {"urls": [], "domains": [], "doc_count": 0, "last_updated": None}
        urls = set(x["url"] for x in d)
        domains = set(x["metadata"].get("source", "") for x in d)
        count = len(d)
        lt = [x["metadata"].get("crawled_at", None) for x in d if x["metadata"].get("crawled_at")]
        if not lt:
            return {"urls": list(urls), "domains": list(domains), "doc_count": 0, "last_updated": None}
        mx = max(lt)
        dt = datetime.fromisoformat(mx.replace("Z", "+00:00"))
        tz = datetime.now().astimezone().tzinfo
        dt = dt.astimezone(tz)
        last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        return {"urls": list(urls), "domains": list(domains), "doc_count": count, "last_updated": last_updated}
    except Exception as e:
        st.error(f"DB Stats error: {e}")
        return None

#############################
# UI Progress
#############################
def init_progress_state():
    if "processing_urls" not in st.session_state:
        st.session_state.processing_urls = []
    if "progress_placeholder" not in st.session_state:
        st.session_state.progress_placeholder = st.sidebar.empty()

def add_processing_url(url: str):
    norm_url = normalize_url(url)
    if norm_url not in st.session_state.processing_urls:
        st.session_state.processing_urls.append(norm_url)
    update_progress()

def remove_processing_url(url: str):
    norm_url = normalize_url(url)
    if norm_url in st.session_state.processing_urls:
        st.session_state.processing_urls.remove(norm_url)
    update_progress()

def update_progress():
    unique_urls = list(dict.fromkeys(st.session_state.get("processing_urls", [])))
    content = "### Currently Processing URLs:\n" + "\n".join(f"- {url}" for url in unique_urls)
    st.session_state.progress_placeholder.markdown(content)

#############################
# Main Streamlit App
#############################
async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System (Supabase)", page_icon="🤖", layout="wide")
    init_progress_state()

    if "chunk_max_chars" not in st.session_state:
        st.session_state.chunk_max_chars = 5000
    if "n_matches" not in st.session_state:
        st.session_state.n_matches = 5
    if "max_snippet_len" not in st.session_state:
        st.session_state.max_snippet_len = 500
    if "crawl_delay" not in st.session_state:
        st.session_state.crawl_delay = 1.0
    if "crawl_word_threshold" not in st.session_state:
        st.session_state.crawl_word_threshold = 50
    if "use_js_for_crawl" not in st.session_state:
        st.session_state.use_js_for_crawl = False
    if "rate_limiter_base_delay_min" not in st.session_state:
        st.session_state.rate_limiter_base_delay_min = 1.0
    if "rate_limiter_base_delay_max" not in st.session_state:
        st.session_state.rate_limiter_base_delay_max = 2.0
    if "rate_limiter_max_delay" not in st.session_state:
        st.session_state.rate_limiter_max_delay = 30.0
    if "rate_limiter_max_retries" not in st.session_state:
        st.session_state.rate_limiter_max_retries = 2

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed = set()
        db_stats = get_db_stats()
        if db_stats and db_stats["urls"]:
            st.session_state.urls_processed = set(db_stats["urls"])
            st.session_state.processing_complete = True if db_stats["doc_count"] > 0 else False


    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = None

    update_progress()

    st.title("Dynamic RAG Chat System (Supabase)")
    db_stats = get_db_stats()
    if db_stats and db_stats["doc_count"] > 0:
        st.session_state.processing_complete = True
        st.session_state.urls_processed = set(db_stats["urls"])

    if db_stats and db_stats["doc_count"] > 0:
        st.success("💡 System is ready with existing knowledge base (Supabase)!")
        with st.expander("Knowledge Base Information", expanded=True):
            st.markdown(f"""
### Current Knowledge Base Stats:
- **Documents**: {db_stats['doc_count']}
- **Sources**: {len(db_stats['domains'])}
- **Last updated**: {db_stats['last_updated']}

### Sources include:
{', '.join(db_stats['domains'])}
""")
    else:
        st.info("👋 Welcome! Start by adding a website to create your knowledge base.")

    config_expander = st.expander("Advanced Configuration", expanded=False)
    with config_expander:
        st.subheader("Configuration")
        st.number_input("Chunk Max Characters", min_value=1000, max_value=10000, value=st.session_state.chunk_max_chars, step=500, key="chunk_max_chars", help="Maximum characters per chunk for text splitting.")
        st.number_input("Retrieval Matches Count", min_value=1, max_value=10, value=st.session_state.n_matches, step=1, key="n_matches", help="Number of documents to retrieve for context.")
        st.number_input("Max Snippet Length", min_value=100, max_value=1000, value=st.session_state.max_snippet_len, step=50, key="max_snippet_len", help="Maximum length of the reference snippet in chat.")
        st.checkbox("Use JavaScript for Crawl", value=st.session_state.use_js_for_crawl, key="use_js_for_crawl", help="Enable JavaScript execution during crawling (can be slower).")
        st.number_input("Crawl Delay (seconds)", min_value=0.0, max_value=5.0, value=st.session_state.crawl_delay, step=0.1, key="crawl_delay", help="Delay between page requests during crawling.")
        st.number_input("Crawl Word Threshold", min_value=10, max_value=200, value=st.session_state.crawl_word_threshold, step=10, key="crawl_word_threshold", help="Minimum word count for a text block to be considered for indexing.")
        st.subheader("Rate Limiter Configuration")
        st.number_input("Rate Limiter Base Delay Min (seconds)", min_value=0.1, max_value=5.0, value=st.session_state.rate_limiter_base_delay_min, step=0.1, key="rate_limiter_base_delay_min", help="Minimum base delay for rate limiter.")
        st.number_input("Rate Limiter Base Delay Max (seconds)", min_value=0.1, max_value=10.0, value=st.session_state.rate_limiter_base_delay_max, step=0.1, key="rate_limiter_base_delay_max", help="Maximum base delay for rate limiter.")
        st.number_input("Rate Limiter Max Delay (seconds)", min_value=10.0, max_value=60.0, value=st.session_state.rate_limiter_max_delay, step=5.0, key="rate_limiter_max_delay", help="Maximum delay the rate limiter will reach.")
        st.number_input("Rate Limiter Max Retries", min_value=1, max_value=5, value=st.session_state.rate_limiter_max_retries, step=1, key="rate_limiter_max_retries", help="Maximum retries for rate-limited requests.")


    max_concurrent = st.slider("Max concurrent URLs", min_value=1, max_value=50, value=10)
    follow_links_recursively = st.checkbox("Follow links recursively", value=True)

    ic, cc = st.columns([1, 2])
    with ic:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process.")
        url_input = st.text_input("Website URL", key="url_input", placeholder="example.com or https://example.com")
        if url_input:
            parsed = urlparse(url_input)
            if "sitemap.xml" in url_input:
                pv = url_input
            elif parsed.path in ("", "/"):
                pv = f"{url_input.rstrip('/')}" + "/sitemap.xml"
            else:
                pv = url_input
            st.caption(f"Will try: {pv}")

        c1, c2 = st.columns(2)
        with c1:
            pb = st.button("Process URL", disabled=st.session_state.is_processing)
        with c2:
            if st.button("Clear Database", disabled=st.session_state.is_processing):
                delete_all_chunks()

        if pb and url_input:
            if normalize_url(url_input) not in set(normalize_url(u) for u in st.session_state.urls_processed): # Normalize URLs for comparison
                st.session_state.is_processing = True
                progress_status = st.empty()
                progress_status.info("Discovering URLs and starting crawl...")

                with st.spinner("Discovery & Parallel Crawl..."):
                    fu = pv
                    found = get_urls_from_sitemap(fu)
                    if not found:
                        progress_status.warning(f"Sitemap not found or empty at {fu}. Falling back to input URL.")
                        found = [url_input]
                    else:
                        progress_status.success(f"Sitemap URLs found: {len(found)}")

                    if follow_links_recursively:
                        discovered = await discover_internal_links(found, max_depth=9)
                        progress_status.success(f"Discovered {len(discovered)} internal URLs.")
                    else:
                        discovered = found

                    if discovered:
                        await crawl_parallel(discovered, max_concurrent=max_concurrent)
                        progress_status.success(f"Crawling and indexing completed for {len(discovered)} URLs.")
                    else:
                        progress_status.warning("No URLs to crawl after discovery.")

                progress_status.empty()
                st.session_state.urls_processed.update(discovered)
                st.session_state.processing_complete = True
                st.session_state.is_processing = False
                st.rerun()
            else:
                st.warning("This URL has already been processed!")

        if st.session_state.urls_processed:
            st.subheader("Processed URLs:")
            up = list(st.session_state.urls_processed)
            for x in up[:3]:
                st.write(f"✓ {x}")
            remaining = len(up) - 3
            if remaining > 0:
                st.write(f"_...and {remaining} more_")
                with st.expander("Show all URLs"):
                    for uxx in up[3:]:
                        st.write(f"✓ {uxx}")

    with cc:
        if st.session_state.processing_complete:
            st.subheader("Chat Interface")
            for m in st.session_state.messages:
                role = "user" if m.get("role") == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(m["content"], unsafe_allow_html=True)

            user_query = st.chat_input("Ask a question about the processed content...")
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query, unsafe_allow_html=True)
                combined_context = retrieve_relevant_documents(user_query, n_matches=st.session_state.n_matches, max_snippet_len=st.session_state.max_snippet_len)
                sys = ("You have access to the following context:\n" + combined_context + "\nAnswer the question.")
                msgs = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user_query}
                ]
                try:
                    r = await openai_client.chat.completions.create(
                        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                        messages=msgs
                    )
                    a = r.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": a})
                    with st.chat_message("assistant"):
                        st.markdown(a, unsafe_allow_html=True)
                        with st.expander("References"):
                            st.markdown(combined_context, unsafe_allow_html=True)
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"Error: {e}", unsafe_allow_html=True)

            if st.button("Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("Please process a URL first to start chatting!")

    st.markdown("---")
    db_stats_display = get_db_stats()
    if db_stats_display and db_stats["doc_count"] > 0:
        st.markdown(f"System Status: 🟢 Ready with {db_stats_display['doc_count']} documents from {len(db_stats_display['domains'])} sources")
    else:
        st.markdown("System Status: 🟡 Waiting for content")

if __name__ == "__main__":
    asyncio.run(main())
