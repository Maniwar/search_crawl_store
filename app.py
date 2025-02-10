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

# --- Helper Functions --- (No changes - but defined here for app.py scope)
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
import utils
from utils import get_urls_from_sitemap, format_sitemap_url, same_domain

# --- Optimized Crawler Config ---
from utils import get_crawler_config

# --- Document Processing & Storage ---
from utils import extract_title_and_summary_from_markdown, process_chunk, insert_chunk_to_supabase_batch, process_and_store_document

# --- Database and Stats Functions --- (No changes)
from utils import delete_all_chunks, get_db_stats

# --- UI Progress functions --- (No changes)
from utils import init_progress_state, add_processing_url, remove_processing_url, update_progress
from utils import discover_internal_links, crawl_parallel # Ensure these are imported from utils

# --- Callback functions for UI elements ---
def update_use_js_crawl():
    st.session_state.use_js_for_crawl = st.session_state.checkbox_use_js_for_crawl_value
def update_follow_links_recursively():
    st.session_state.follow_links_recursively = st.session_state.checkbox_follow_links_recursive_value
def clear_database_button_callback(): # Callback for clear database button
    delete_all_chunks()

# --- Main Streamlit App ---
async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System", page_icon="🤖", layout="wide", menu_items={'About': "RAG System - Crawl, Store, & Chat"})

    if "initial_setup_done" not in st.session_state:
        install_playwright()
        init_progress_state()
        for key, default_value in {
            "chunk_max_chars": 3800, "n_matches": 4, "max_snippet_len": 450, "crawl_delay": 0.4, "crawl_word_threshold": 60,
            "use_js_for_crawl": False, "rate_limiter_base_delay_min": 0.4, "rate_limiter_base_delay_max": 1.2,
            "rate_limiter_max_delay": 15.0, "rate_limiter_max_retries": 2, "messages": [], "processing_complete": False,
            "urls_processed": set(), "is_processing": False, "suggested_questions": None, "max_concurrent": 25,
            "follow_links_recursively": False,
            "check_robots_txt": False, # Initialize check_robots_txt here
            "url_include_patterns": "", "url_exclude_patterns": ""
        }.items():
            st.session_state[key] = default_value
        db_stats = get_db_stats()
        if db_stats and db_stats["urls"]:
            st.session_state.urls_processed.update(db_stats["urls"])
            st.session_state.processing_complete = db_stats["doc_count"] > 0
        st.session_state.initial_setup_done = True

    update_progress()

    st.title("Dynamic RAG Chat System")
    if st.session_state.processing_complete and (db_stats := get_db_stats()):
        st.success(f"Knowledge base ready ({db_stats['doc_count']} docs, {len(db_stats['domains'])} sources)!")
        if st.expander("Knowledge Base Stats", expanded=False):
            st.markdown(f"""**Documents**: {db_stats['doc_count']}\n**Sources**: {len(db_stats['domains'])}
**Last updated**: {db_stats['last_updated']}\n\n**Sources:**\n{', '.join(db_stats['domains'])}""")
    else:
        st.info("👋 Add website URLs to build your knowledge base.")

    with st.sidebar:
        st.header("Configuration")
        st.session_state.max_concurrent = st.slider("Concurrent URLs", 1, 50, value=st.session_state.max_concurrent, step=5, help="Number of URLs to crawl in parallel for speed.")
        st.checkbox(
            "Follow Links Recursively",
            value=st.session_state.follow_links_recursively,
            key="checkbox_follow_links_recursive_value",
            help="Crawl internal links recursively.",
            on_change=update_follow_links_recursively # Use defined callback function
        )
        with st.expander("Advanced Settings"):
            st.session_state.chunk_max_chars = st.number_input("Chunk Size (Characters)", 1000, 8000, value=st.session_state.chunk_max_chars, step=500, help="Text chunk size for processing.")
            st.session_state.n_matches = st.slider("Retrieval Matches (Chat)", 1, 7, value=st.session_state.n_matches, step=1, help="Number of documents retrieved for chat context.")
            st.session_state.max_snippet_len = st.slider("Snippet Length (Chat)", 100, 1200, value=st.session_state.max_snippet_len, step=100, help="Length of reference snippets in chat.")
            st.session_state.crawl_delay = st.slider("Crawl Delay (Seconds)", 0.0, 3.0, value=st.session_state.crawl_delay, step=0.1, format="%.1f", help="Delay between requests to avoid overloading websites.")
            st.session_state.crawl_word_threshold = st.slider("Word Threshold (Indexing)", 10, 150, value=st.session_state.crawl_word_threshold, step=10, help="Minimum words for indexing text blocks.")
            st.checkbox(
                "Enable JavaScript Rendering",
                value=st.session_state.use_js_for_crawl,
                key="checkbox_use_js_for_crawl_value",
                help="Render dynamic content, but crawling will be slower.",
                on_change=update_use_js_crawl # Use defined callback function
            )
            st.subheader("Rate Limiter")
            st.session_state.rate_limiter_base_delay_min = st.slider("Base Delay (Min Sec)", 0.1, 3.0, value=st.session_state.rate_limiter_base_delay_min, step=0.1, format="%.1f")
            st.session_state.rate_limiter_base_delay_max = st.slider("Base Delay (Max Sec)", 0.1, 7.0, value=st.session_state.rate_limiter_base_delay_max, step=0.1, format="%.1f")
            st.session_state.rate_limiter_max_delay = st.slider("Max Delay (Sec)", 5.0, 45.0, value=st.session_state.rate_limiter_max_delay, step=5.0, format="%.0f")
            st.session_state.rate_limiter_max_retries = st.slider("Max Retries", 1, 4, value=st.session_state.rate_limiter_max_retries, step=1)
            st.subheader("Crawl Rules")
            st.session_state.check_robots_txt = st.checkbox("Respect robots.txt", value=st.session_state.check_robots_txt, key="check_robots_txt", help="Enable robots.txt compliance (recommended).")
            st.session_state.url_include_patterns = st.text_area("Include URLs matching pattern (one per line)", value=st.session_state.url_include_patterns, height=70, key="url_include_patterns", help="Crawl only URLs that match these patterns (leave empty to include all).")
            st.session_state.url_exclude_patterns = st.text_area("Exclude URLs matching pattern (one per line)", value=st.session_state.url_exclude_patterns, height=70, key="url_exclude_patterns", help="Exclude URLs that match these patterns (leave empty to exclude none).")

        st.button("Clear Knowledge Base", on_click=clear_database_button_callback, disabled=st.session_state.is_processing) # Use callback for button

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

                if st.session_state.follow_links_recursively:
                    status_placeholder.info(f"Following internal links (max depth: {st.session_state.get('max_depth_discover_links', 2)})...")
                    discovered_urls = await discover_internal_links(crawl_urls, max_depth=2) # Call function directly (imported from utils)
                    urls_to_crawl = discovered_urls
                    status_placeholder.success(f"Discovered {len(urls_to_crawl)} URLs.")
                else:
                    status_placeholder.info("Recursive link following: OFF")

                if urls_to_crawl:
                    status_placeholder.info(f"Crawling and indexing {len(urls_to_crawl)} pages...")
                    await crawl_parallel(urls_to_crawl, max_concurrent=st.session_state.max_concurrent) # Call function directly (imported from utils)
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
                rag_context = retrieve_relevant_documents(user_question, n_matches=st.session_state.n_matches, max_snippet_len=st.session_state.max_snippet_len)
                rag_prompt_enhanced = f"Context:\n{rag_context}\n\nQuestion: {user_question}\n\nAnswer:"
                messages_openai = [{"role": "system", "content": "Answer user questions concisely and accurately based on the context provided."}, {"role": "user", "content": rag_prompt_enhanced}]

                try:
                    openai_response = await openai_client.chat.completions.create(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), messages=messages_openai)
                    ai_answer = openai_response.choices[0].message.content.strip()
                    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                    with st.chat_message("assistant"):
                        st.markdown(ai_answer, unsafe_allow_html=True)
                        with st.expander("References"):
                            st.markdown(rag_context, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Chat interface error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I can't answer right now."})
                    st.chat_message("assistant").markdown("Sorry, I can't answer right now.")

            if st.button("Clear Chat History", type="secondary", key="clear_chat_history"):
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("Process website content to enable chat.")

    st.markdown("---")
    status_display_final = get_db_stats()
    if status_display_final and status_display_final["doc_count"] > 0:
        st.markdown(f"**Status:** 🟢 Ready | **Docs:** {status_display_final['doc_count']} | **Sources:** {len(status_display_final['domains'])}")
    else:
        st.markdown("**Status:** 🟡 Waiting for content")

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    asyncio.run(main())
