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
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin, urlunparse, unquote
from xml.etree import ElementTree
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
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
nltk_data_path = os.path.join(".", "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

print(f"NLTK Data Path: {nltk_data_path}")

try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt data already found.")
except LookupError:
    print("NLTK punkt data not found, downloading...")
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('taggers/punkt_tab')
    print("NLTK punkt_tab data already found.")
except LookupError:
    print("NLTK punkt_tab data not found, downloading...")
    nltk.download('punkt_tab', download_dir=nltk_data_path)

os.environ['NLTK_DATA'] = nltk_data_path

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
js_click_all = """(async () => { const clickable = document.querySelectorAll("a, button"); for (let el of clickable) { try { el.click(); await new Promise(r => setTimeout(r, 300)); } catch(e) {} } })();"""

# --- Helper Functions ---
def normalize_url(u: str) -> str:
    parts = urlparse(u)
    normalized_path = parts.path.rstrip('/') if parts.path != '/' else parts.path
    normalized = parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower(), path=normalized_path)
    return urlunparse(normalized)

def chunk_text(t: str, max_chars: int = 3800) -> List[str]:
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

# --- OpenAI Embedding ---
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

def cosine_similarity(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def extract_reference_snippet(content: str, query: str, snippet_length: int = 250) -> str:
    sentences = sent_tokenize(content)
    query_embedding = asyncio.run(get_embedding(query))
    if query_embedding is None: return "Error generating embedding for query snippet."
    best_sentence = ""
    max_similarity = -1
    for sentence in sentences:
        sentence_embedding = asyncio.run(get_embedding(sentence))
        if sentence_embedding:
            similarity = cosine_similarity(query_embedding, sentence_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentence
    snippet_to_highlight = best_sentence if best_sentence else content[:snippet_length]
    def highlight_word(word):
        if word.lower().startswith("http"):
            return word
        for term in query.split():
            if re.search(re.escape(term), word, flags=re.IGNORECASE):
                return f'<span style="background-color: red; color: white; font-weight:bold;">{word}</span>'
        return word

    highlighted = " ".join(highlight_word(w) for w in snippet_to_highlight.split())
    return highlighted

def retrieve_relevant_documentation(query: str, n_matches: int = 3, max_snippet_len: int = 400) -> str:
    e = asyncio.run(get_embedding(query))
    r = supabase.rpc("match_documents", {"query_embedding": e, "match_count": n_matches * 2}).execute()
    d = r.data
    if not d:
        return "No relevant documentation found."

    snippets = []
    for doc in d[:n_matches]:
        raw_url = doc['url']
        print(f"Raw URL from DB: {raw_url}")
        content_slice = doc["content"][:max_snippet_len]
        snippet = extract_reference_snippet(content_slice, query, max_snippet_len // 2)
        cleaned_url = unquote(raw_url)
        print(f"URL after unquote: {cleaned_url}")
        cleaned_url = normalize_url(cleaned_url)
        print(f"URL after normalize_url: {cleaned_url}")
        cleaned_url = cleaned_url.rstrip('%3C/%3E').rstrip('<//>')
        print(f"URL after aggressive rstrip: {cleaned_url}")
        snippets.append(f"""\n#### {doc['title']}\n\n{snippet}\n\n**Source:** [{doc['metadata']['source']}]({cleaned_url})\nSimilarity: {doc['similarity']:.2f}""")

    return "\n".join(snippets)

# --- Sitemap Helpers ---
def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    urls = []
    try:
        print(f"Fetching sitemap from: {sitemap_url}")  # Debug log
        response = requests.get(sitemap_url, timeout=15)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"Sitemap status code: {response.status_code}")  # Debug log
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        url_elements = root.findall('ns:url/ns:loc', namespace)
        if not url_elements:
            print("Warning: No <loc> elements found in sitemap XML.")  # Debug log
        for url_element in url_elements:
            urls.append(url_element.text.strip())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap: {e}")
    except ElementTree.ParseError as e:
        print(f"Error parsing sitemap XML: {e} - Sitemap URL: {sitemap_url}")
    except Exception as e:
        print(f"Unexpected error in get_urls_from_sitemap: {e}")
    return urls

def format_sitemap_url(url: str) -> str:
    if "sitemap.xml" in url:
        return url
    parsed = urlparse(url)
    if parsed.path in ("", "/"):
        return f"{url.rstrip('/')}" + "/sitemap.xml"
    return url

def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc

# --- Optimized Crawler Config ---
def get_crawler_config(st_session_state) -> CrawlerRunConfig:
    browser_config = BrowserConfig(
        use_js=st_session_state.get("use_js_for_crawl", False),
        js_snippets=[js_click_all]
    )
    rate_limiter = RateLimiter(
        base_delay_range=(st_session_state.get("rate_limiter_base_delay_min", 0.4), st_session_state.get("rate_limiter_base_delay_max", 1.2)),
        max_delay=st.session_state.get("rate_limiter_max_delay", 15.0),
        max_retries=st.session_state.get("rate_limiter_max_retries", 2)
    )
    crawler_config = CrawlerRunConfig(
        browser_config=browser_config,
        cache_mode=CacheMode.AUTO,
        rate_limiter=rate_limiter,
        crawler_monitor=CrawlerMonitor(display_mode=DisplayMode.SILENT)
    )
    return crawler_config

# --- Document Processing & Storage ---
def extract_title_and_summary_from_markdown(markdown_content: str) -> dict:
    title_match = re.search(r"^#\s+(.*)", markdown_content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "No Title Found"
    sentences = sent_tokenize(markdown_content)
    summary = sentences[0] if sentences else "No summary available."
    return {"title": title, "summary": summary}

async def process_chunk(url: str, chunk_id: int, content: str, metadata: dict, embedding_model):
    if not content.strip():
        return None

    title_summary = extract_title_and_summary_from_markdown(content) if metadata.get('filetype') == 'markdown' else {'title': metadata.get('title', 'No Title'), 'summary': ''}
    embedding_vector = await embedding_model(content)

    if embedding_vector is None:
        print(f"Warning: Embedding generation failed for chunk {chunk_id} from {url}")
        return None

    return {
        'url': url,
        'chunk_id': chunk_id,
        'content': content,
        'embedding': embedding_vector,
        'metadata': metadata,
        'title': title_summary['title'],
        'summary': title_summary['summary']
    }

async def insert_chunk_to_supabase_batch(chunk_batch: List[dict]):
    if not chunk_batch:
        return

    try:
        data, count = supabase.table("rag_chunks").insert(chunk_batch).execute()
        if count:
            print(f"Inserted {count} chunks in batch.")
        else:
            print("Warning: No chunks inserted in this batch.")
    except Exception as e:
        print(f"Error inserting chunk batch to Supabase: {e}")


async def process_and_store_document(url: str, content: str, metadata: dict, chunk_max_chars: int, crawl_word_threshold: int, embedding_model):
    chunks = chunk_text(content, chunk_max_chars)
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk.split()) > crawl_word_threshold: # Apply word count threshold here
            processed_chunk_info = await process_chunk(url, i, chunk, metadata, embedding_model)
            if processed_chunk_info: # Check if process_chunk returned a valid chunk
                processed_chunks.append(processed_chunk_info)

    if processed_chunks: # Only insert if there are processed chunks
        try:
            await insert_chunk_to_supabase_batch(processed_chunks)
            print(f"Indexed {len(processed_chunks)} chunks from {url}")
        except Exception as e:
            print(f"Error storing document chunks for {url}: {e}")
    else:
        print(f"Skipped indexing {url} due to word count threshold or no valid content.")


# --- Database and Stats Functions --- (No changes)
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

# --- UI Progress functions --- (No changes)
def init_progress_state():
    if "processing_urls" not in st.session_state:
        st.session_state.processing_urls = []
    if "progress_placeholder" not in st.session_state:
        st.session_state.progress_placeholder = st.sidebar.empty()
    if 'discover_links_progress' not in st.session_state: # Add progress for link discovery
        st.session_state.discover_links_progress = 0.0
    if 'crawl_index_progress' not in st.session_state: # Add progress for crawl/index
        st.session_state.crawl_index_progress = 0.0

def update_progress_discover_links(processed_urls, message):
    total_urls = st.session_state.get("total_urls_to_discover", 0)
    if total_urls > 0:
        progress_percentage = (processed_urls / total_urls)
        st.session_state.discover_links_progress = progress_percentage
        st.session_state.progress_placeholder.progress(st.session_state.discover_links_progress, text=f"{message} ({processed_urls}/{total_urls})")
    else:
        st.session_state.progress_placeholder.info("Link discovery started...")

def update_progress_crawl_index(processed_urls, total_urls, message):
    if total_urls > 0:
        progress_percentage = (processed_urls / total_urls)
        st.session_state.crawl_index_progress = progress_percentage
        st.session_state.progress_placeholder.progress(st.session_state.crawl_index_progress, text=f"{message} ({processed_urls}/{total_urls})")
    else:
        st.session_state.progress_placeholder.info("Crawling and Indexing started...")

def add_processing_url(url):
    norm_url = normalize_url(url)
    if norm_url not in st.session_state.processing_urls:
        st.session_state.processing_urls.append(norm_url)
    update_progress()

def remove_processing_url(url):
    norm_url = normalize_url(url)
    if norm_url in st.session_state.processing_urls:
        st.session_state.processing_urls.remove(norm_url)
    update_progress()

def update_progress():
    discover_progress = st.session_state.get("discover_links_progress", 0.0)
    crawl_index_progress = st.session_state.get("crawl_index_progress", 0.0)
    overall_progress = (discover_progress + crawl_index_progress) / 2.0

    if overall_progress > 0:
        st.session_state.progress_placeholder.progress(overall_progress)
    else:
        unique_urls = list(dict.fromkeys(st.session_state.get("processing_urls", [])))
        content = "### URLs in Queue:\n" + "\n".join(f"- {url}" for url in unique_urls)
        st.session_state.progress_placeholder.markdown(content)

# --- Main Streamlit App --- (No changes except passing st_session_state to crawling functions)
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
        st.info("👋 Welcome! Start by adding a website URLs to build your knowledge base.")

    max_concurrent = st.slider("Max concurrent URLs", 1, 50, value=st.session_state.get("max_concurrent", 25), step=5, help="Number of URLs to crawl in parallel for speed.")
    follow_links_recursively = st.checkbox("Follow Links Recursively", value=st.session_state.get("follow_links_recursively", False), key="checkbox_follow_links_recursive_value", help="Crawl internal links recursively.")
    use_js_for_crawl = st.checkbox("Enable JavaScript Rendering", value=st.session_state.get("use_js_for_crawl", False), key="checkbox_use_js_for_crawl_value", help="Render dynamic content, but crawling will be slower.")
    chunk_max_chars = st.slider("Chunk Size (Characters)", 1000, 8000, value=st.session_state.get("chunk_max_chars", 3800), step=500, help="Text chunk size for processing.")
    crawl_delay = st.slider("Crawl Delay (Seconds)", 0.0, 3.0, value=st.session_state.get("crawl_delay", 0.4), step=0.1, format="%.1f", help="Delay between requests to avoid overloading websites.")
    crawl_word_threshold = st.slider("Word Threshold (Indexing)", 10, 150, value=st.session_state.get("crawl_word_threshold", 60), step=10, help="Minimum words for indexing text blocks.")
    rate_limiter_base_delay_min = st.slider("Base Delay (Min Sec)", 0.1, 3.0, value=st.session_state.get("rate_limiter_base_delay_min", 0.4), step=0.1, format="%.1f")
    rate_limiter_base_delay_max = st.slider("Base Delay (Max Sec)", 0.1, 7.0, value=st.session_state.get("rate_limiter_base_delay_max", 1.2), step=0.1, format="%.1f")
    rate_limiter_max_delay = st.slider("Max Delay (Sec)", 5.0, 45.0, value=st.session_state.get("rate_limiter_max_delay", 15.0), step=5.0, format="%.0f")
    rate_limiter_max_retries = st.slider("Max Retries", 1, 4, value=st.session_state.get("rate_limiter_max_retries", 2), step=1)
    check_robots_txt = st.checkbox("Respect robots.txt", value=st.session_state.get("check_robots_txt", False), key="check_robots_txt", help="Enable robots.txt compliance (recommended).")
    url_include_patterns = st.text_area("Include URLs matching pattern (one per line)", value=st.session_state.get("url_include_patterns", ""), height=70, key="url_include_patterns", help="Crawl only URLs that match these patterns (leave empty to include all).")
    url_exclude_patterns = st.text_area("Exclude URLs matching pattern (one per line)", value=st.session_state.get("url_exclude_patterns", ""), height=70, key="url_exclude_patterns", help="Exclude URLs that match these patterns (leave empty to exclude none).")


    st.button("Clear Knowledge Base", on_click=clear_database_button_callback, disabled=st.session_state.is_processing)

    input_col, chat_col = st.columns([1, 2])
    with input_col:
        st.subheader("Add Website Content")
        website_url = st.text_input("Enter Website or Sitemap URL", placeholder="https://example.com", help="Input URL to crawl; will attempt to find sitemap and respect crawl rules.")
        if website_url:
            st.caption(f"Sitemap URL (if found): `{format_sitemap_url(website_url)}`")
        process_website_button = st.button("Process Website Content", disabled=st.session_state.is_processing)

        if process_website_button and website_url:
            normalized_website_url = normalize_url(website_url)
            if normalized_website_url not in st.session_state.urls_processed:
                st.session_state.is_processing = True
                status_placeholder = st.empty()
                status_placeholder.info(f"Processing: {normalized_website_url}...")

                sitemap_urls = get_urls_from_sitemap(format_sitemap_url(website_url))
                crawl_urls = sitemap_urls if sitemap_urls else [website_url]
                st.session_state.total_urls_to_discover = len(crawl_urls) # Initialize total URLs to discover

                include_patterns = st.session_state.url_include_patterns.strip().splitlines() if st.session_state.url_include_patterns else None
                exclude_patterns = st.session_state.url_exclude_patterns.strip().splitlines() if st.session_state.url_exclude_patterns else None

                def filter_urls_by_pattern(urls_to_filter: List[str], patterns: List[str], exclude=False) -> List[str]:
                    if not patterns: return urls_to_filter
                    filtered_urls = []
                    for url_item in urls_to_filter:
                        for pattern in patterns:
                            if re.search(pattern, url_item):
                                if not exclude:
                                    filtered_urls.append(url_item)
                                break
                        else:
                            if exclude:
                                filtered_urls.append(url_item)
                    return filtered_urls

                if include_patterns:
                    status_placeholder.info("Applying URL inclusion patterns...")
                    crawl_urls = filter_urls_by_pattern(crawl_urls, include_patterns)
                    status_placeholder.success(f"URLs after inclusion filtering: {len(crawl_urls)}.")
                if exclude_patterns:
                    status_placeholder.info("Applying URL exclusion patterns...")
                    crawl_urls = filter_urls_by_pattern(crawl_urls, exclude_patterns, exclude=True)
                    status_placeholder.success(f"URLs after exclusion filtering: {len(crawl_urls)}.")


                urls_to_crawl = crawl_urls

                if st.session_state.follow_links_recursively:
                    status_placeholder.info(f"Following internal links (max depth: {st.session_state.get('max_depth_discover_links', 2)})...")
                    discovered_urls = await discover_internal_links(st.session_state, crawl_urls, max_depth=2) # Pass st.session_state
                    urls_to_crawl = list(discovered_urls)
                    status_placeholder.success(f"Discovered {len(urls_to_crawl)} URLs.")
                else:
                    status_placeholder.info("Recursive link following: OFF")

                if urls_to_crawl:
                    status_placeholder.info(f"Crawling and indexing {len(urls_to_crawl)} pages...")
                    st.session_state.total_urls_to_crawl_index = len(urls_to_crawl)
                    await crawl_parallel(st.session_state, urls_to_crawl, max_concurrent=st.session_state.get("max_concurrent", 25)) # Pass st.session_state
                    status_placeholder.success(f"Crawling & indexing complete. Knowledge base updated!")
                    st.session_state.urls_processed.add(normalized_website_url)
                    st.session_state.processing_complete = True
                else:
                    status_placeholder.warning("No URLs found to crawl after filtering.")

                status_placeholder.empty()
                st.session_state.is_processing = False
                st.rerun()
            else:
                st.warning("This URL has already been processed. Please add a new URL.")

        if st.session_state.urls_processed:
            st.subheader("Processed URLs")
            for url in sorted(list(st.session_state.urls_processed))[:5]:
                st.write(f"✓ {url}")
            if st.checkbox(f"Show all {len(st.session_state.urls_processed)} URLs"):
                for url in sorted(list(st.session_state.urls_processed))[5:]:
                    st.write(f"✓ {url}")

    with chat_col:
        if st.session_state.processing_complete:
            st.subheader("Chat Interface")
            for msg in st.session_state.messages:
                role_str = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role_str):
                    st.markdown(msg["content"], unsafe_allow_html=True)

            user_question = st.chat_input("Ask me anything...")
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question, unsafe_allow_html=True)
                rag_context = retrieve_relevant_documentation(user_question, n_matches=3, max_snippet_len=st.session_state.get("max_snippet_len", 600))
                sys = f"You have access to the following context snippets:\n{rag_context}\n---\nAnswer the question based on this context:\n"
                msgs = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user_question}
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
                            st.markdown(rag_context, unsafe_allow_html=True)
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
    db_stats = get_db_stats()
    if db_stats and db_stats["doc_count"] > 0:
        st.markdown(f"**Status:** 🟢 Ready | **Docs:** {db_stats.get('doc_count', 'N/A')} | **Sources:** {len(db_stats.get('domains', [])) if db_stats else 'N/A'}")
    else:
        st.markdown("**Status:** 🟡 Waiting for content")

if __name__ == "__main__":
    asyncio.run(main())
