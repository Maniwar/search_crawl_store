import nest_asyncio
nest_asyncio.apply()
import os
import subprocess
import sys

# Install Playwright without requiring sudo.
def install_playwright():
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

install_playwright()

import asyncio
import json
import requests
import re
import streamlit as st
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional # Keep Optional import
from urllib.parse import urlparse, urljoin, urlunparse, unquote # Import unquote
from xml.etree import ElementTree
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize # Ensure this line is present
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

# NLTK data path setup - download locally if needed
nltk_data_path = os.path.join(".", "nltk_data")  # Define local nltk_data directory
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)  # Create directory if it doesn't exist

print(f"NLTK Data Path: {nltk_data_path}") # Debug: Print NLTK data path

# --- NLTK Data Download and Setup (Simplified - Run ONCE at startup) ---
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt data already found.")
except LookupError:
    print("NLTK punkt data not found, downloading punkt...")
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('taggers/punkt_tab')
    print("NLTK punkt_tab data already found.")
except LookupError:
    print("NLTK punkt_tab data not found, downloading punkt_tab...")
    nltk.download('punkt_tab', download_dir=nltk_data_path)

os.environ['NLTK_DATA'] = nltk_data_path # Set NLTK_DATA environment variable
# --- End of Simplified NLTK Setup ---


load_dotenv()

# Environment variables and client setup
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in your environment variables.")

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

#############################
# Helper Functions (No Changes)
#############################

def normalize_url(u: str) -> str:
    parts = urlparse(u)
    normalized_path = parts.path.rstrip('/') if parts.path != '/' else parts.path
    normalized = parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower(), path=normalized_path)
    return urlunparse(normalized)

def chunk_text(t: str, max_chars: int = 3800) -> List[str]: # Reduced chunk size for potentially better snippets
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
# OpenAI Embedding (Modified extract_reference_snippet) - No Changes
#############################
async def get_embedding(text: str) -> List[float]: # No change
    try:
        r = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return r.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 1536

def cosine_similarity(vec1, vec2): # Added cosine_similarity function
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def extract_reference_snippet(content: str, query: str, snippet_length: int = 250) -> str: # Modified - Highlight color changed to red
    sentences = sent_tokenize(content) # Use sentence tokenizer
    query_embedding = asyncio.run(get_embedding(query)) # Get embedding of the query
    if query_embedding is None: return "Error generating embedding for query snippet."
    best_sentence = ""
    max_similarity = -1
    for sentence in sentences: # Iterate through sentences
        sentence_embedding = asyncio.run(get_embedding(sentence)) # Embed each sentence
        if sentence_embedding:
            similarity = cosine_similarity(query_embedding, sentence_embedding) # Calculate cosine similarity
            if similarity > max_similarity: # Find the most similar sentence
                max_similarity = similarity
                best_sentence = sentence
    snippet_to_highlight = best_sentence if best_sentence else content[:snippet_length] # Fallback to content slice if no best sentence
    def highlight_word(word): # Highlighting function - changed highlight color to red
        if word.lower().startswith("http"):
            return word
        for term in query.split():
            if re.search(re.escape(term), word, flags=re.IGNORECASE):
                return f'<span style="background-color: red; color: white; font-weight:bold;">{word}</span>' # Changed to red, white text, bold
        return word

    highlighted = " ".join(highlight_word(w) for w in snippet_to_highlight.split())
    return highlighted

def retrieve_relevant_documentation(query: str, n_matches: int = 3, max_snippet_len: int = 400) -> str: # Modified - added n_matches and max_snippet_len
    e = asyncio.run(get_embedding(query))
    r = supabase.rpc("match_documents", {"query_embedding": e, "match_count": n_matches * 2}).execute() # Fetch more initially
    d = r.data
    if not d:
        return "No relevant documentation found."

    snippets = []
    for doc in d[:n_matches]: # Use top n_matches documents
        raw_url = doc['url']
        print(f"Raw URL from DB: {raw_url}") # Debug print 1: URL from DB

        content_slice = doc["content"][:max_snippet_len] # Slice content for snippet extraction
        snippet = extract_reference_snippet(content_slice, query, max_snippet_len // 2) # Extract snippet

        cleaned_url = unquote(raw_url) # Decode URL-encoded characters
        print(f"URL after unquote: {cleaned_url}") # Debug print 2: After unquote

        cleaned_url = normalize_url(cleaned_url) # Clean and normalize URL
        print(f"URL after normalize_url: {cleaned_url}") # Debug print 3: After normalize_url

        cleaned_url = cleaned_url.rstrip('%3C/%3E').rstrip('<//>') # Aggressively remove suffix if present
        print(f"URL after aggressive rstrip: {cleaned_url}") # Debug print 4: After rstrip

        snippets.append(f"""\n#### {doc['title']}\n\n{snippet}\n\n**Source:** [{doc['metadata']['source']}]({cleaned_url})\nSimilarity: {doc['similarity']:.2f}""") # Use cleaned_url

    return "\n".join(snippets) # Return combined snippets


#############################
# Sitemap Helpers (Fixed get_urls_from_sitemap)
#############################

def get_urls_from_sitemap(u: str) -> List[str]:
    try:
        print(f"Fetching sitemap from: {u}") # Debug: Log URL being fetched
        r = requests.get(u)
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"Sitemap fetched successfully. Status code: {r.status_code}") # Debug: Log status code
        xml_content = r.content
        ro = ElementTree.fromstring(xml_content)
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text.strip() for loc in ro.findall(".//ns:loc", ns)]
        print(f"Found {len(urls)} URLs in sitemap.") # Debug: Log number of URLs found
        return urls
    except requests.exceptions.RequestException as e:
        print(f"Sitemap fetch error for {u}: {e}") # Debug: Detailed request error
        return []
    except ElementTree.ParseError as e:
        print(f"Sitemap XML Parse error for {u}: {e}") # Debug: XML parsing error
        return []
    except Exception as e:
        print(f"Sitemap processing error for {u}: {e}") # Debug: General error
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
# Code from Crawl4AI Docs (No Changes - corrected session_state already)
#############################

def get_run_config(st_session_state, with_js: bool = False) -> CrawlerRunConfig:
    kwargs = {
        "cache_mode": CacheMode.BYPASS,
        "stream": False,
        "exclude_external_links": False,
        "wait_for_images": True,
        "delay_before_return_html": 1.0,
        "excluded_tags": ["header", "footer", "nav", "aside"],
        "word_count_threshold": st_session_state.get("crawl_word_threshold", 50),
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

#############################
# BFS Link Discovery (optional) (Modified discover_internal_links with Debugging)
#############################
async def discover_internal_links(st_session_state, start_urls: List[str], max_depth: int = 3) -> List[str]:
    visited = set()
    discovered = set()
    queue = deque((url, 0) for url in start_urls)

    bc = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    run_conf = get_run_config(st_session_state, with_js=st_session_state.get("use_js_for_crawl", False))

    async with AsyncWebCrawler(config=bc) as crawler:
        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            print(f"discover_internal_links: Dequeued URL: {url}, Depth: {depth}") # Debug: URL dequeued

            # BFS approach: just parse links, don't store content
            run_config_instance = get_run_config(st_session_state, with_js=st_session_state.get("use_js_for_crawl", False))
            print(f"discover_internal_links: Calling crawler.arun for URL: {url}") # Debug: Before crawler.arun
            result = await crawler.arun(url=url, config=run_config_instance)
            print(f"discover_internal_links: crawler.arun returned for URL: {url}, Success: {result.success}") # Debug: After crawler.arun

            if result.success:
                discovered.add(url)
                links_dict = getattr(result, "links", {})
                internal_links = links_dict.get("internal", [])
                print(f"discover_internal_links: Found {len(internal_links)} internal links on {url}") # Debug: Links found
                for link_obj in internal_links:
                    href = link_obj.get("href")
                    if href:
                        abs_url = urljoin(url, href)
                        if abs_url and same_domain(abs_url, url):
                            queue.append((abs_url, depth + 1))
            else:
                print(f"discover_internal_links: Link discovery failed for {url}: {result.error_message}") # Debug: Crawl failure

    return list(discovered)

#############################
# Parallel Crawl (arun_many) (No Changes - corrected session_state already)
#############################
async def crawl_parallel(st_session_state, urls: List[str], max_concurrent: int = 10, use_js: bool = False):
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
        rate_limiter=RateLimiter(
            base_delay=(st_session_state.get("rate_limiter_base_delay_min", 1.0), st.session_state.get("rate_limiter_base_delay_max", 2.0)),
            max_delay=st.session_state.get("rate_limiter_max_delay", 30.0),
            max_retries=st.session_state.get("rate_limiter_max_retries", 2),
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
    run_conf = get_run_config(st_session_state, with_js=use_js)

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

#############################
# DB & Stats (No Changes)
#############################
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

#############################
# UI Progress (No Changes)
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
# Main Streamlit App (No Changes - corrected session_state already)
#############################
async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System (Supabase)", page_icon="🤖", layout="wide")
    init_progress_state()

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
    if "crawl_word_threshold" not in st.session_state: # Initialize word threshold
        st.session_state.crawl_word_threshold = 50
    if "use_js_for_crawl" not in st.session_state: # Initialize use_js_for_crawl
        st.session_state.use_js_for_crawl = False
    if "rate_limiter_base_delay_min" not in st.session_state: # Initialize rate limiter settings
        st.session_state.rate_limiter_base_delay_min = 1.0
    if "rate_limiter_base_delay_max" not in st.session_state:
        st.session_state.rate_limiter_base_delay_max = 2.0
    if "rate_limiter_max_delay" not in st.session_state:
        st.session_state.rate_limiter_max_delay = 30.0
    if "rate_limiter_max_retries" not in st.session_state:
        st.session_state.rate_limiter_max_retries = 2


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

    max_concurrent = st.slider("Max concurrent URLs", min_value=1, max_value=50, value=st.session_state.get("max_concurrent", 10)) # Corrected: use .get
    follow_links_recursively = st.checkbox("Follow links recursively", value=st.session_state.get("follow_links_recursively", True)) # Corrected: use .get
    use_js_for_crawl = st.checkbox("Use Javascript for Crawling", value=st.session_state.get("use_js_for_crawl", False))

    ic, cc = st.columns([1, 2])
    with ic:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process.")
        url_input = st.text_input("Website URL", key="url_input", placeholder="example.com or https://example.com")
        # Attempt to guess a sitemap if user didn't provide
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
                st.session_state.processing_complete = False
                st.session_state.urls_processed = set()
                st.session_state.messages = []
                st.session_state.suggested_questions = None
                st.success("Database cleared successfully!")
                st.rerun()

        if pb and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing = True
                with st.spinner("Discovery & Parallel Crawl..."):
                    # Phase 0: try to get sitemap or fallback
                    fu = pv  # from above
                    print(f"Attempting to fetch sitemap from URL: {fu}") # Debug: log sitemap URL attempt
                    found_sitemap_urls = get_urls_from_sitemap(fu) # Renamed to be more explicit
                    print(f"URLs found from sitemap: {found_sitemap_urls}") # Debug: log URLs found
                    if found_sitemap_urls: # Check if sitemap URLs were found (not just if list is empty/None)
                        urls_to_crawl = found_sitemap_urls # Use sitemap URLs if available
                        print("Sitemap URLs found. Proceeding with sitemap URLs.") # Debug log
                    else:
                        urls_to_crawl = [url_input] # Fallback to initial URL if no sitemap
                        print("No valid sitemap URLs found. Falling back to initial URL for crawling.") # Debug log

                    # BFS link discovery if user wants recursion
                    if follow_links_recursively:
                        print(f"Follow links recursively is enabled. Starting link discovery from: {urls_to_crawl}") # Debug message
                        discovered = await discover_internal_links(st.session_state, urls_to_crawl, max_depth=9) # Use urls_to_crawl here
                        print(f"Discovered URLs after recursive link discovery: {len(discovered)}") # Debug message
                    else:
                        discovered = urls_to_crawl # Use either sitemap urls or initial url directly
                        print("Follow links recursively is disabled. Using initial URLs (or sitemap URLs if found).") # Debug message

                    # Now do a parallel crawl in batch
                    await crawl_parallel(st.session_state, discovered, max_concurrent=max_concurrent, use_js=use_js_for_crawl) # Pass use_js_for_crawl

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
            if st.checkbox(f"Show all {len(up)} URLs"):
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
                rag_context = retrieve_relevant_documentation(user_query, n_matches=3, max_snippet_len=600) # Call with n_matches and max_snippet_len
                sys = f"You have access to the following context snippets:\n{rag_context}\n---\nAnswer the question based on this context:\n"
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
                        with st.expander("References"): # Expander to show all references
                            st.markdown(rag_context, unsafe_allow_html=True) # Show full RAG context with snippets
                except Exception as e:
                    st.error(f"Chat interface error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                    st.chat_message("assistant").markdown(f"Error: {e}", unsafe_allow_html=True)

            if st.button("Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("Please process a URL first to start chatting!")

    st.markdown("---")
    db_stats = get_db_stats() # Get db_stats here to have it available for status display
    if db_stats and db_stats["doc_count"] > 0:
        st.markdown(f"**Status:** 🟢 Ready | **Docs:** {db_stats['doc_count']} | **Sources:** {len(db_stats['domains'])}") # Use db_stats here
    else:
        st.markdown("**Status:** 🟡 Waiting for content")

if __name__ == "__main__":
    asyncio.run(main())
