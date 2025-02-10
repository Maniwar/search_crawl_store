import nest_asyncio
nest_asyncio.apply()
import os
import subprocess
import sys

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
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI
from nltk.tokenize import sent_tokenize  # Import NLTK sentence tokenizer

from crawl4ai import (AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, RateLimiter, CrawlerMonitor, DisplayMode)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from collections import deque

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

# --- Helper Functions --- (normalize_url, chunk_text remain same)
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

async def get_embedding(text: str) -> Optional[List[float]]: # get_embedding remains same
    try:
        response = await openai_client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding
    except Exception as error:
        st.error(f"Embedding error: {error}")
        return None

# --- Optimized Semantic Snippet Extraction ---
def extract_reference_snippet(content: str, query: str, snippet_length: int = 250) -> str:
    sentences = sent_tokenize(content) # Tokenize content into sentences
    query_embedding = asyncio.run(get_embedding(query)) # Get embedding for query

    if query_embedding is None: # Handle embedding error
        return "Error generating embedding for query snippet."

    best_sentence = ""
    max_similarity = -1

    for sentence in sentences:
        sentence_embedding = asyncio.run(get_embedding(sentence))
        if sentence_embedding: # Ensure sentence embedding was successful
            similarity = cosine_similarity(query_embedding, sentence_embedding) # Calculate cosine similarity
            if similarity > max_similarity:
                max_similarity = similarity
                best_sentence = sentence

    snippet_to_highlight = best_sentence if best_sentence else content[:snippet_length] # Fallback to beginning of content if no sentence found

    def highlight_word(word): # Highlighting function remains the same
        if word.lower().startswith("http"): return word
        for term in query.split():
            if re.search(re.escape(term), word, flags=re.IGNORECASE):
                return f'<span style="background-color: yellow; font-weight:bold;">{word}</span>'
        return word

    highlighted_snippet = " ".join(highlight_word(word) for word in snippet_to_highlight.split())
    return highlighted_snippet

def cosine_similarity(vec1, vec2): # Cosine similarity calculation function
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = sum(x ** 2 for x in vec1) ** 0.5
    magnitude2 = sum(x ** 2 for x in vec2) ** 0.5
    if not magnitude1 or not magnitude2:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# --- Optimized RAG retrieval --- (No changes needed)
retrieve_relevant_documents = retrieve_relevant_documents # No changes needed

# --- Sitemap Helpers --- (No changes needed)
get_urls_from_sitemap = get_urls_from_sitemap # No changes needed
format_sitemap_url = format_sitemap_url # No changes needed
same_domain = same_domain # No changes needed

# --- Optimized Crawler Config --- (No changes needed)
get_crawler_config = get_crawler_config # No changes needed

# --- Document Processing & Storage --- (No changes needed)
extract_title_and_summary_from_markdown = extract_title_and_summary_from_markdown # No changes needed
process_chunk = process_chunk # No changes needed
insert_chunk_to_supabase_batch = insert_chunk_to_supabase_batch # No changes needed
process_and_store_document = process_and_store_document # No changes needed

# --- Optimized Crawling Functions --- (No changes needed)
crawl_parallel = crawl_parallel # No changes needed
discover_internal_links = discover_internal_links # No changes needed

# --- Database and Stats Functions --- (No changes needed)
delete_all_chunks = delete_all_chunks # No changes needed
get_db_stats = get_db_stats # No changes needed

# --- UI Progress functions --- (No changes needed)
init_progress_state = init_progress_state # No changes needed
add_processing_url = add_processing_url # No changes needed
remove_processing_url = remove_processing_url # No changes needed
update_progress = update_progress # No changes needed

# --- Main Streamlit App --- (No changes needed)
async def main(): # No changes needed
    st.set_page_config(page_title="Dynamic RAG Chat System", page_icon="🤖", layout="wide", menu_items={'About': "RAG System - Crawl, Store, & Chat"})

    if "initial_setup_done" not in st.session_state:
        install_playwright()
        init_progress_state()
        for key, default_value in {
            "chunk_max_chars": 3800, "n_matches": 4, "max_snippet_len": 450, "crawl_delay": 0.4, "crawl_word_threshold": 60,
            "use_js_for_crawl": False, "rate_limiter_base_delay_min": 0.4, "rate_limiter_base_delay_max": 1.2,
            "rate_limiter_max_delay": 15.0, "rate_limiter_max_retries": 2, "messages": [], "processing_complete": False,
            "urls_processed": set(), "is_processing": False, "suggested_questions": None, "max_concurrent": 25,
            "follow_links_recursively": False
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
        st.success(f"Knowledge base ready! ({db_stats['doc_count']} docs, {len(db_stats['domains'])} sources)!")
        if st.expander("Knowledge Base Stats", expanded=False):
            st.markdown(f"""**Documents**: {db_stats['doc_count']}\n**Sources**: {len(db_stats['domains'])}
**Last updated**: {db_stats['last_updated']}\n\n**Sources:**\n{', '.join(db_stats['domains'])}""")
    else:
        st.info("👋 Add website URLs to build your knowledge base.")

    with st.sidebar:
        st.header("Configuration")
        st.session_state.max_concurrent = st.slider("Concurrent URLs", 1, 50, value=st.session_state.max_concurrent, step=5, help="Number of URLs to crawl in parallel for speed.")
        st.session_state.follow_links_recursively = st.checkbox("Recursive Crawling", value=st.session_state.follow_links_recursively, help="Crawl and index internal links for deeper content.")
        with st.expander("Advanced Settings"):
            st.session_state.chunk_max_chars = st.number_input("Chunk Size (Characters)", 1000, 8000, value=st.session_state.chunk_max_chars, step=500, help="Text chunk size for processing.")
            st.session_state.n_matches = st.slider("Retrieval Matches (Chat)", 1, 7, value=st.session_state.n_matches, step=1, help="Number of documents retrieved for chat context.")
            st.session_state.max_snippet_len = st.slider("Snippet Length (Chat)", 100, 1200, value=st.session_state.max_snippet_len, step=100, help="Length of reference snippets in chat.")
            st.session_state.crawl_delay = st.slider("Crawl Delay (Seconds)", 0.0, 3.0, value=st.session_state.crawl_delay, step=0.1, format="%.1f", help="Delay between requests to avoid overloading websites.")
            st.session_state.crawl_word_threshold = st.slider("Word Threshold (Indexing)", 10, 150, value=st.session_state.crawl_word_threshold, step=10, help="Minimum words for indexing text blocks.")
            st.session_state.use_js_for_crawl = st.checkbox("Enable JavaScript Rendering", value=st.session_state.use_js_for_crawl, help="Render dynamic content, but crawling will be slower.")
            st.subheader("Rate Limiter")
            st.session_state.rate_limiter_base_delay_min = st.slider("Base Delay (Min Sec)", 0.1, 3.0, value=st.session_state.rate_limiter_base_delay_min, step=0.1, format="%.1f")
            st.session_state.rate_limiter_base_delay_max = st.slider("Base Delay (Max Sec)", 0.1, 7.0, value=st.session_state.rate_limiter_base_delay_max, step=0.1, format="%.1f")
            st.session_state.rate_limiter_max_delay = st.slider("Max Delay (Sec)", 5.0, 45.0, value=st.session_state.rate_limiter_max_delay, step=5.0, format="%.0f")
            st.session_state.rate_limiter_max_retries = st.slider("Max Retries", 1, 4, value=st.session_state.rate_limiter_max_retries, step=1)
        if st.button("Clear Knowledge Base", on_click=delete_all_chunks, disabled=st.session_state.is_processing):
            pass

    input_col, chat_col = st.columns([1, 2])
    with input_col:
        st.subheader("Add Website Content")
        website_url = st.text_input("Enter Website or Sitemap URL", placeholder="https://example.com", help="Input URL to crawl; will attempt to find sitemap.")
        if website_url:
            st.caption(f"Sitemap URL (if found): `{format_sitemap_url(website_url)}`")
        process_website_button = st.button("Process Website Content", disabled=st.session_state.is_processing)

        if process_website_button and website_url:
            normalized_website_url = normalize_url(website_url)
            if normalized_website_url not in set(normalize_url(u) for u in st.session_state.urls_processed):
                st.session_state.is_processing = True
                status_placeholder = st.empty()
                status_placeholder.info(f"Processing: {normalized_website_url}...")

                sitemap_urls = get_urls_from_sitemap(format_sitemap_url(website_url))
                crawl_urls = sitemap_urls if sitemap_urls else [website_url]

                if st.session_state.follow_links_recursively:
                    status_placeholder.info(f"Following internal links (max depth: {st.session_state.get('max_depth_discover_links', 2)})...")
                    discovered_urls = await discover_internal_links(crawl_urls, max_depth=2)
                    urls_to_crawl = discovered_urls
                    status_placeholder.success(f"Discovered {len(urls_to_crawl)} URLs.")
                else:
                    status_placeholder.info("Recursive link following: OFF")

                if urls_to_crawl:
                    status_placeholder.info(f"Crawling and indexing {len(urls_to_crawl)} pages...")
                    await crawl_parallel(urls_to_crawl, max_concurrent=st.session_state.max_concurrent)
                    status_placeholder.success(f"Crawling & indexing complete. Knowledge base updated!")
                    st.session_state.urls_processed.update(urls_to_crawl)
                    st.session_state.processing_complete = True
                else:
                    status_placeholder.warning("No URLs found to crawl.")

                status_placeholder.empty()
                st.session_state.is_processing = False
                st.rerun()
            else:
                st.warning("This URL has already been processed. Add a new one!")

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
    nltk.download('punkt') # Download punkt tokenizer for sentence splitting
    asyncio.run(main())
