import nest_asyncio
nest_asyncio.apply()
import os
os.system('playwright install --with-deps chromium')
import asyncio
import json
import requests
import re
import streamlit as st
from datetime import datetime, timezone
from typing import List, Dict, Any
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

load_dotenv()

# Environment variables and client setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in your environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- Define JS snippet (if needed) ---
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

# --- Helper Functions ---

def normalize_url(u: str) -> str:
    """
    Normalize a URL by lowercasing the scheme and netloc,
    and stripping any trailing slash (except for the root path).
    """
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

async def get_embedding(text: str) -> List[float]:
    try:
        r = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return r.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 1536

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

def retrieve_relevant_documentation(query: str) -> str:
    e = asyncio.run(get_embedding(query))
    r = supabase.rpc("match_documents", {"query_embedding": e, "match_count": 5}).execute()
    d = r.data
    if not d:
        return "No relevant documentation found."
    best = max(d, key=lambda x: x.get("similarity", 0))
    snippet = extract_reference_snippet(best["content"], query)
    return f"\n# {best['title']}\n\n{snippet}\n...\nSource: {best['url']}\nSimilarity: {best['similarity']:.3f}\n"

def get_urls_from_sitemap(u: str) -> List[str]:
    try:
        r = requests.get(u)
        r.raise_for_status()
        ro = ElementTree.fromstring(r.content)
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text.strip() for loc in ro.findall(".//ns:loc", ns)]
    except Exception as e:
        print(f"Sitemap error: {e}")
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

def get_run_config(with_js: bool = False) -> CrawlerRunConfig:
    kwargs = {
        "cache_mode": CacheMode.BYPASS,
        "stream": False,
        "exclude_external_links": False,
        "wait_for_images": True,
        "delay_before_return_html": 1.0,
        "excluded_tags": ["header", "footer", "nav", "aside"],
        "word_count_threshold": 50
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

async def process_chunk(chunk: str, num: int, url: str) -> Dict[str, Any]:
    ts = extract_title_and_summary_from_markdown(chunk)
    embedding = await get_embedding(ts["summary"])
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
    try:
        supabase.table("rag_chunks").upsert(d).execute()
    except Exception as e:
        print(f"Insert error: {e}")

async def process_and_store_document(url: str, md: str):
    chunks = chunk_text(md)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk_to_supabase(item) for item in processed_chunks]
    await asyncio.gather(*insert_tasks)

# --- UI Progress Widget (Sidebar) ---

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

# --- Advanced Parallel Crawling ---
async def crawl_parallel(urls: List[str], max_concurrent: int = 10):
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 2.0),
            max_delay=30.0,
            max_retries=2,
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
    run_conf = get_run_config(with_js=False)
    async with AsyncWebCrawler(config=bc) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=run_conf,
            dispatcher=dispatcher
        )
        for r in results:
            if r.success:
                md = r.markdown_v2.raw_markdown
                await process_and_store_document(r.url, md)
            else:
                print(f"Failed crawling {r.url}: {r.error_message}")

# --- Recursive Crawl with Global Concurrency Limit ---
async def recursive_crawl(url: str, max_depth: int = 9, current_depth: int = 0, 
                          processed: set = None, sema: asyncio.Semaphore = None):
    if processed is None:
        processed = set()
    if sema is None:
        sema = asyncio.Semaphore(10)  # Global concurrency limit of 10
    if current_depth > max_depth or url in processed:
        return
    processed.add(url)
    norm_url = normalize_url(url)
    add_processing_url(norm_url)
    try:
        await asyncio.sleep(1)
        async with sema:
            dispatcher = MemoryAdaptiveDispatcher(
                memory_threshold_percent=85.0,
                check_interval=1.0,
                max_session_permit=5,
                monitor=None
            )
            bc = BrowserConfig(
                headless=True,
                verbose=False,
                extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
            )
            run_conf = get_run_config(with_js=False)
            async with AsyncWebCrawler(config=bc) as crawler:
                result = await crawler.arun(url=url, config=run_conf)
                if result.success:
                    print(f"Crawled: {url} (depth {current_depth})")
                    md = result.markdown_v2.raw_markdown
                    await process_and_store_document(url, md)
                    links_dict = getattr(result, "links", {})
                    internal_links = links_dict.get("internal", [])
                    tasks = []
                    for link in internal_links:
                        href = link.get("href")
                        absolute_url = urljoin(url, href) if href else None
                        if absolute_url and same_domain(absolute_url, url) and absolute_url not in processed:
                            tasks.append(recursive_crawl(absolute_url, max_depth, current_depth + 1, processed, sema))
                    if tasks:
                        await asyncio.gather(*tasks)
                else:
                    print(f"Error crawling {url}: {result.error_message}")
    finally:
        remove_processing_url(norm_url)

def delete_all_chunks():
    supabase.table("rag_chunks").delete().neq("id", "").execute()

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
            return {"urls": list(urls), "domains": list(domains), "doc_count": count, "last_updated": None}
        mx = max(lt)
        dt = datetime.fromisoformat(mx.replace("Z", "+00:00"))
        tz = datetime.now().astimezone().tzinfo
        dt = dt.astimezone(tz)
        last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        return {"urls": list(urls), "domains": list(domains), "doc_count": count, "last_updated": last_updated}
    except Exception as e:
        print(f"DB Stats error: {e}")
        return None

# --- Main Streamlit App ---
async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System (Supabase)", page_icon="🤖", layout="wide")
    init_progress_state()  # Initialize sidebar state
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed = set()
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = None
    
    update_progress()  # Initial update
    
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
    
    max_concurrent = st.slider("Max concurrent URLs", min_value=1, max_value=50, value=10)
    follow_links_recursively = st.checkbox("Follow links recursively", value=True)
    
    ic, cc = st.columns([1, 2])
    with ic:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process.")
        url_input = st.text_input("Website URL", key="url_input", placeholder="example.com or https://example.com")
        if url_input:
            if "sitemap.xml" in url_input:
                pv = url_input
            else:
                parsed = urlparse(url_input)
                if parsed.path in ("", "/"):
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
                st.session_state.processing_complete = False
                st.session_state.urls_processed = set()
                st.session_state.messages = []
                st.session_state.suggested_questions = None
                st.success("Database cleared successfully!")
                st.rerun()
        if pb and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing = True
                with st.spinner("Crawling & Processing..."):
                    if "sitemap.xml" in url_input:
                        found = get_urls_from_sitemap(url_input)
                    else:
                        fu = format_sitemap_url(url_input)
                        found = get_urls_from_sitemap(fu)
                        if not found:
                            found = [url_input]
                    sema = asyncio.Semaphore(max_concurrent)
                    if follow_links_recursively:
                        await recursive_crawl(found[0], max_depth=9, sema=sema)
                    else:
                        await crawl_parallel(found, max_concurrent=max_concurrent)
                st.session_state.urls_processed.add(url_input)
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
                dr = retrieve_relevant_documentation(user_query)
                sys = "You have access to the following context:\n" + dr + "\nAnswer the question."
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
                            st.markdown(dr, unsafe_allow_html=True)
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
    if db_stats and db_stats["doc_count"] > 0:
        st.markdown(f"System Status: 🟢 Ready with {db_stats['doc_count']} documents from {len(db_stats['domains'])} sources")
    else:
        st.markdown("System Status: 🟡 Waiting for content")

if __name__ == "__main__":
    asyncio.run(main())
