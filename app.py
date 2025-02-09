import nest_asyncio
nest_asyncio.apply()
import os
os.system('playwright install chromium')
os.system('playwright install-deps chromium')
import asyncio
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

# Crawl4AI imports
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

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_KEY]):
    st.error("Missing environment variables")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = AsyncOpenAI(api_key=OPENAI_KEY)

# --- Core Functions ---
def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip('/') if parsed.path != '/' else parsed.path
    ).geturl()

def chunk_text(content: str, chunk_size: int = 5000) -> List[str]:
    paragraphs = content.split('\n\n')
    chunks = []
    current = []
    current_len = 0
    
    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size:
            chunks.append('\n\n'.join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len
            
    if current:
        chunks.append('\n\n'.join(current))
    return chunks

# --- Progress Tracking ---
def init_session():
    if "processing" not in st.session_state:
        st.session_state.processing = set()
    if "progress_ph" not in st.session_state:
        st.session_state.progress_ph = st.sidebar.empty()

def update_progress():
    urls = st.session_state.processing
    content = "## Active Crawls\n" + "\n".join(f"- {url}" for url in urls)
    st.session_state.progress_ph.markdown(content)

def add_url(url: str):
    norm = normalize_url(url)
    if norm not in st.session_state.processing:
        st.session_state.processing.add(norm)
        update_progress()

def remove_url(url: str):
    norm = normalize_url(url)
    if norm in st.session_state.processing:
        st.session_state.processing.remove(norm)
        update_progress()

# --- Crawling System ---
def get_browser_config():
    return BrowserConfig(
        browser="chromium",
        headless=True,
        args=[
            "--no-sandbox",
            "--single-process",
            "--disable-dev-shm-usage",
            "--disable-gpu"
        ]
    )

async def crawl_page(url: str):
    try:
        add_url(url)
        crawler = AsyncWebCrawler(config=get_browser_config())
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=100
            )
        )
        
        if result.success:
            await store_content(url, result.markdown_v2.raw_markdown)
            return result.links.get("internal", [])
        return []
    
    finally:
        remove_url(url)

async def recursive_crawl(
    start_url: str,
    max_depth: int = 2,
    semaphore: asyncio.Semaphore = None
):
    sem = semaphore or asyncio.Semaphore(6)
    processed = set()
    
    async def _crawl(url: str, depth: int):
        if depth > max_depth or url in processed:
            return
            
        async with sem:
            processed.add(url)
            links = await crawl_page(url)
            
            if depth < max_depth:
                child_links = [urljoin(url, l.get("href")) for l in links]
                await asyncio.gather(*[
                    _crawl(link, depth+1)
                    for link in child_links
                    if same_domain(link, start_url)
                ])
    
    await _crawl(start_url, 0)

def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc

# --- Database Operations ---
async def store_content(url: str, content: str):
    try:
        chunks = chunk_text(content)
        embeddings = await asyncio.gather(*[
            get_embedding(chunk) for chunk in chunks
        ])
        
        records = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            records.append({
                "id": f"{url}-{idx}",
                "url": url,
                "content": chunk,
                "embedding": emb,
                "metadata": {
                    "chunk_num": idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": urlparse(url).netloc
                }
            })
        
        supabase.table("rag_chunks").upsert(records).execute()
        
    except Exception as e:
        st.error(f"Storage error: {str(e)}")

async def get_embedding(text: str) -> List[float]:
    resp = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

# --- UI Components ---
def main():
    st.set_page_config("Smart Crawler", "🤖", "wide")
    init_session()
    
    st.title("AI-Powered Website Crawler")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Crawling Controls")
        url = st.text_input("Start URL", key="crawl_url")
        depth = st.slider("Crawl Depth", 1, 3, 1)
        
        if st.button("Start Crawl"):
            if url:
                st.session_state.processing.clear()
                asyncio.create_task(recursive_crawl(url, depth))
            else:
                st.warning("Enter a valid URL")
        
        if st.button("Clear Database"):
            supabase.table("rag_chunks").delete().neq("id", "").execute()
            st.rerun()
    
    with col2:
        st.header("Processing Status")
        if st.session_state.processing:
            st.info(f"Active crawls: {len(st.session_state.processing)}")
            st.write(list(st.session_state.processing)[:5])
            if len(st.session_state.processing) > 5:
                st.write(f"...and {len(st.session_state.processing)-5} more")
        else:
            st.info("No active crawls")
        
        st.header("Chat Interface")
        # Add chat components here

if __name__ == "__main__":
    asyncio.run(main())
