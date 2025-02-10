import nest_asyncio
nest_asyncio.apply()
import os
import subprocess
import sys
import functools  # Import functools for callback

# Optimized Playwright installation
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
from xml.etree.ElementTree import ElementTree
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from supabase import create_client, Client
from openai import AsyncOpenAI

# Advanced imports for Crawl4AI
from crawl4ai import (AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, RateLimiter, CrawlerMonitor, DisplayMode)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from collections import deque

nltk.download('punkt') # Ensure punkt tokenizer is downloaded
load_dotenv()

# --- Setup: Environment variables and clients ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    st.error("Please set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in your environment variables.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# --- JS snippet --- (No changes)
js_click_all = """(async () => { const clickable = document.querySelectorAll("a, button"); for (let el of clickable) { try { el.click(); await new Promise(r => setTimeout(r, 150)); } catch(e) {} } })();"""

# --- Helper Functions --- (No changes)
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

async def get_embedding(text: str) -> Optional[List[float]]:
    try:
        response = await openai_client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding
    except Exception as error:
        st.error(f"Embedding error: {error}")
        return None

def cosine_similarity(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# --- Optimized Semantic Snippet Extraction --- (No changes)
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
        if word.lower().startswith("http"): return word
        for term in query.split():
            if re.search(re.escape(term), word, flags=re.IGNORECASE):
                return f'<span style="background-color: yellow; font-weight:bold;">{word}</span>'
        return word
    highlighted_snippet = " ".join(highlight_word(word) for word in snippet_to_highlight.split())
    return highlighted_snippet

# --- Optimized RAG retrieval --- (No changes)
def retrieve_relevant_documents(query: str, n_matches: int, max_snippet_len: int) -> str:
    embedding_vector = asyncio.run(get_embedding(query))
    if embedding_vector is None:
        return "Error generating embedding for the query."

    res = supabase.rpc('match_documents', {'query_embedding': embedding_vector, 'match_count': n_matches}).execute()
    if not res.data:
        return "No relevant documentation found."

    snippets = []
    for doc in res.data:
        content_slice = doc["content"][:max_snippet_len]
        snippet = extract_reference_snippet(content_slice, query, max_snippet_len // 2)
        snippets.append(f"""\n#### {doc['title']}\n\n{snippet}\n\n**Source:** [{doc['metadata']['source']}]({doc['url']})\nSimilarity: {doc['similarity']:.2f}""")
    return "\n".join(snippets)

# --- Sitemap Helpers ---
def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Fetches URLs from a sitemap XML."""
    urls = []
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for url_element in root.findall('ns:url/ns:loc', namespace):
            urls.append(url_element.text.strip())
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap: {e}")
    except ElementTree.ParseError as e:
        print(f"Error parsing sitemap XML: {e}")
    return urls

def format_sitemap_url(url: str) -> str:
    """Attempts to create a sitemap URL from a given website URL."""
    parsed_url = urlparse(url)
    base_url = parsed_url.scheme + "://" + parsed_url.netloc
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    return line.split(":", 1)[1].strip() # Return sitemap from robots.txt if found
    except requests.exceptions.RequestException:
        pass # Ignore errors trying to fetch robots.txt, use default sitemap.xml

    return sitemap_url # Fallback to default sitemap.xml if robots.txt fails or no sitemap in it

def same_domain(url1: str, url2: str) -> bool:
    """Checks if two URLs are on the same domain."""
    return urlparse(url1).netloc == urlparse(url2).netloc

# --- Optimized Crawler Config ---
def get_crawler_config(st_session_state) -> CrawlerRunConfig:
    """Generates CrawlerRunConfig from Streamlit session state."""
    browser_config = BrowserConfig(
        use_js=st_session_state.get("use_js_for_crawl", False), # Corrected to use .get
        js_snippets=[st_session_state.get("js_click_all", "")], # Corrected to use .get, and ensure js_click_all is in session_state if needed
    )
    rate_limiter = RateLimiter(
        base_delay_range=(st_session_state.get("rate_limiter_base_delay_min", 0.4), st_session_state.get("rate_limiter_base_delay_max", 1.2)), # Corrected to use .get
        max_delay=st_session_state.get("rate_limiter_max_delay", 15.0), # Corrected to use .get
        max_retries=st_session_state.get("rate_limiter_max_retries", 2) # Corrected to use .get
    )
    crawler_config = CrawlerRunConfig(
        browser_config=browser_config,
        cache_mode=CacheMode.AUTO,
        rate_limiter=rate_limiter,
        crawler_monitor=CrawlerMonitor(display_mode=DisplayMode.SILENT) # Or DisplayMode.STREAMLIT for UI integration
    )
    return crawler_config

# --- Document Processing & Storage ---
def extract_title_and_summary_from_markdown(markdown_content: str) -> dict:
    """Extracts title and summary from markdown content."""
    title_match = re.search(r"^#\s+(.*)", markdown_content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "No Title Found"
    sentences = sent_tokenize(markdown_content)
    summary = sentences[0] if sentences else "No summary available."
    return {"title": title, "summary": summary}

async def process_chunk(url: str, chunk_id: int, content: str, metadata: dict, embedding_model) -> Optional[dict]:
    """Processes a text chunk: extracts metadata and generates embedding."""
    if not content.strip():
        return None  # Skip empty chunks

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

async def insert_chunk_to_supabase_batch(chunk_batch: List[dict]) -> None:
    """Inserts a batch of processed chunks into Supabase."""
    if not chunk_batch:
        return

    try:
        data, count = supabase.table("documents").insert(chunk_batch).execute()
        if count:
            print(f"Inserted {count} chunks in batch.")
        else:
            print("Warning: No chunks inserted in this batch.")
    except Exception as e:
        print(f"Error inserting chunk batch to Supabase: {e}")


async def process_and_store_document(url: str, content: str, metadata: dict, chunk_max_chars: int, crawl_word_threshold: int, embedding_model) -> int:
    """Processes a document, chunks it, and stores it in Supabase."""
    chunks = []
    word_count = 0
    total_chunks_indexed = 0

    for text_chunk in chunk_text(content, chunk_max_chars):
        current_word_count = len(text_chunk.split())
        if current_word_count > crawl_word_threshold:
            chunk_metadata = metadata.copy()
            chunk_metadata['source'] = metadata.get('final_url') or url # Prefer final URL if available
            chunk = await process_chunk(url, len(chunks) + 1, text_chunk, chunk_metadata, embedding_model)
            if chunk:
                chunks.append(chunk)
                word_count += current_word_count

    if chunks:
        try:
            await insert_chunk_to_supabase_batch(chunks)
            total_chunks_indexed = len(chunks)
            print(f"Indexed {total_chunks_indexed} chunks, {word_count} words from {url}")
        except Exception as e:
            print(f"Error storing document chunks for {url}: {e}")
    else:
        print(f"Skipped indexing {url} due to word count threshold or processing issues.")

    return total_chunks_indexed


# --- Database and Stats Functions --- (No changes - now defined in app.py)
def delete_all_chunks():
    """Deletes all documents from the Supabase database."""
    try:
        response = supabase.table('documents').delete().neq('url', '').execute()
        print(f"Database cleared. Deleted {response.count} documents.")
        return True
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

def get_db_stats():
    """Fetches database statistics (document count, domain sources)."""
    try:
        res_docs = supabase.table('documents').select('id', count='exact').execute()
        doc_count = res_docs.count if res_docs else 0
        res_domains = supabase.table('documents').select('metadata->>source').execute()
        domains = set()
        if res_domains and res_domains.data:
            for item in res_domains.data:
                source_url = item.get('metadata', {}).get('source')
                if source_url:
                    domains.add(urlparse(source_url).netloc)

        last_updated = None
        if doc_count > 0:
             res_updated = supabase.table('documents').select('created_at').order('created_at', desc=True).limit(1).execute()
             if res_updated and res_updated.data:
                last_updated_str = res_updated.data[0].get('created_at')
                if last_updated_str:
                    last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S UTC')


        return {
            "doc_count": doc_count,
            "domains": list(domains),
            "last_updated": last_updated
        }
    except Exception as e:
        print(f"Error fetching DB stats: {e}")
        return None

# --- UI Progress functions --- (No changes - now defined in app.py)
def init_progress_state():
    """Initializes Streamlit session state for URL processing progress."""
    if 'processing_progress' not in st.session_state:
        st.session_state.processing_progress = 0.0
    if 'processing_url_status' not in st.session_state:
        st.session_state.processing_url_status = {} # url: {status: 'pending'/'processing'/'done'/'error', message: '...'}
    if 'total_urls_to_process' not in st.session_state:
        st.session_state.total_urls_to_process = 0
    if 'processed_urls_count' not in st.session_state:
        st.session_state.processed_urls_count = 0

def add_processing_url(url):
    """Adds a URL to the processing status list."""
    st.session_state.processing_url_status[url] = {'status': 'pending', 'message': 'Added to queue.'}
    st.session_state.total_urls_to_process = len(st.session_state.processing_url_status)
    update_progress()

def remove_processing_url(url, status='done', message='Processed'):
    """Updates and removes a URL from the processing status list."""
    if url in st.session_state.processing_url_status:
        st.session_state.processing_url_status[url]['status'] = status
        st.session_state.processing_url_status[url]['message'] = message
        st.session_state.processed_urls_count = sum(1 for status_item in st.session_state.processing_url_status.values() if status_item['status'] == 'done')
        update_progress()

def update_progress():
    """Updates the overall progress bar in the UI."""
    if st.session_state.total_urls_to_process > 0:
        progress_percentage = (st.session_state.processed_urls_count / st.session_state.total_urls_to_process)
        st.session_state.processing_progress = progress_percentage
    else:
        st.session_state.processing_progress = 0.0

# --- Crawling Functions --- (No changes - now defined in app.py)
async def discover_internal_links(start_urls: List[str], max_depth: int = 2) -> Set[str]:
    """Recursively discovers internal links starting from a list of URLs."""
    config = get_crawler_config(st.session_state)
    crawler = AsyncWebCrawler(crawler_config=config)
    discovered_urls: Set[str] = set()
    queue: deque = deque([(url, 0) for url in start_urls]) # URL and depth

    robots_checked_domains = set() # Track domains to check robots.txt only once if enabled

    while queue:
        current_url, depth = queue.popleft()
        normalized_url = normalize_url(current_url)

        if normalized_url in discovered_urls or depth > max_depth:
            continue

        domain = urlparse(normalized_url).netloc
        if st.session_state.get("check_robots_txt", False) and domain not in robots_checked_domains: # Corrected to use .get
            robots_url = urljoin(current_url, "/robots.txt")
            if not await crawler.is_allowed_by_robots(robots_url, normalized_url):
                print(f" robots.txt disallowed: {normalized_url}")
                robots_checked_domains.add(domain) # Mark domain as checked even if disallowed to avoid repeated checks
                continue # Skip disallowed URL
            robots_checked_domains.add(domain) # Mark domain as checked

        discovered_urls.add(normalized_url)
        print(f"Discovered: {normalized_url} (Depth {depth})")
        add_processing_url(normalized_url) # Add to processing status in UI

        try:
            crawl_result = await crawler.crawl_url(normalized_url)
            if crawl_result and crawl_result.internal_links and depth < max_depth:
                base_url = urlparse(normalized_url).scheme + "://" + urlparse(normalized_url).netloc
                internal_links = {normalize_url(urljoin(base_url, link)) for link in crawl_result.internal_links if same_domain(normalized_url, urljoin(base_url, link))}
                for link in internal_links:
                    if link not in discovered_urls:
                        queue.append((link, depth + 1))
        except Exception as e:
            print(f"Error crawling and discovering links at {normalized_url}: {e}")
            remove_processing_url(normalized_url, status='error', message=f'Crawl error: {e}') # Update UI on error
        else:
             remove_processing_url(normalized_url, status='done', message='Discovered links.') # Update UI on success, even if no new links found.

    await crawler.close_browser() # Ensure browser is closed after discovery
    return discovered_urls


async def crawl_parallel(urls: List[str], max_concurrent: int):
    """Crawls and indexes a list of URLs in parallel with rate limiting and error handling."""
    config = get_crawler_config(st.session_state)
    crawler = AsyncWebCrawler(crawler_config=config)
    dispatcher = MemoryAdaptiveDispatcher(max_concurrency=max_concurrent)
    tasks = {} # Track tasks by url for status updates

    async def crawl_and_index_url(url):
        try:
            crawl_result = await crawler.crawl_url(url)
            if crawl_result and crawl_result.content_text:
                metadata = {
                    'url': url,
                    'final_url': crawl_result.final_url,
                    'http_status': crawl_result.http_status,
                    'headers': str(crawl_result.headers),
                    'filetype': crawl_result.filetype,
                    'title': crawl_result.title or "No Title" # Ensure title is not None
                }
                indexed_chunks = await process_and_store_document(
                    url,
                    crawl_result.content_text,
                    metadata,
                    st.session_state.get("chunk_max_chars", 3800), # Corrected to use .get
                    st.session_state.get("crawl_word_threshold", 60), # Corrected to use .get
                    get_embedding # Assuming get_embedding is defined in app.py or imported
                )
                return indexed_chunks
            else:
                print(f"No content extracted from {url}")
                return 0 # Indicate no chunks indexed

        except Exception as e:
            print(f"Error crawling or indexing {url}: {e}")
            remove_processing_url(url, status='error', message=f'Indexing error: {e}') # Update UI on error
            return 0 # Indicate no chunks indexed due to error
        finally:
            remove_processing_url(url, status='done', message='Indexed.') # Update UI on completion (success or error)


    async def process_url_with_dispatcher(url):
        """Wraps url processing to be dispatched and handle status updates."""
        add_processing_url(url) # Mark as processing in UI
        indexed_count = await crawl_and_index_url(url) # Crawl and index, get chunk count
        return indexed_count # Return chunk count for potential aggregation

    for url in urls:
        normalized_url = normalize_url(url)
        if normalized_url not in st.session_state.urls_processed: # Skip already processed URLs
            task = dispatcher.dispatch(process_url_with_dispatcher(normalized_url))
            tasks[normalized_url] = task # Track task with URL for potential monitoring

    results = await asyncio.gather(*tasks.values()) # Gather results from all dispatched tasks
    total_indexed_chunks = sum(results) # Aggregate total indexed chunks across all URLs
    print(f"Total chunks indexed across all URLs: {total_indexed_chunks}")

    await dispatcher.close()
    await crawler.close_browser() # Close browser after all crawling is done.
    return total_indexed_chunks
