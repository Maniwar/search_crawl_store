import nest_asyncio
nest_asyncio.apply()
import os
import asyncio
import json
import requests
import streamlit as st
from datetime import datetime, timezone
from typing import List, Dict, Any
from urllib.parse import urlparse
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI

# Imports for Crawl4AI and dispatchers
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
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

# --- Begin Helper Functions ---

def chunk_text(t: str, c: int = 5000) -> List[str]:
    r = []
    s = 0
    l = len(t)
    while s < l:
        e = s + c
        if e >= l:
            r.append(t[s:].strip())
            break
        sub = t[s:e]
        cb = sub.rfind("```")
        if cb != -1 and cb > c * 0.3:
            e = s + cb
        elif "\n\n" in sub:
            lb = sub.rfind("\n\n")
            if lb > c * 0.3:
                e = s + lb
        elif ". " in sub:
            lp = sub.rfind(". ")
            if lp > c * 0.3:
                e = s + lp + 1
        sub = t[s:e].strip()
        if sub:
            r.append(sub)
        s = max(s + 1, e)
    return r

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

def retrieve_relevant_documentation(query: str) -> str:
    e = asyncio.run(get_embedding(query))
    r = supabase.rpc("match_documents", {"query_embedding": e, "match_count": 5}).execute()
    d = r.data
    if not d:
        return "No relevant documentation found."
    parts = []
    for row in d:
        parts.append(
            f"\n# {row['title']}\n\n{row['content'][:1000]}\n...\nSource: {row['url']}\nSimilarity: {row['similarity']:.3f}\n"
        )
    return "\n\n---\n\n".join(parts)

def get_urls_from_sitemap(u: str) -> List[str]:
    try:
        r = requests.get(u)
        r.raise_for_status()
        ro = ElementTree.fromstring(r.content)
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text for loc in ro.findall(".//ns:loc", ns)]
    except Exception as e:
        print(f"Sitemap error: {e}")
        return []

def same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc

def format_sitemap_url(u: str) -> str:
    u = u.rstrip("/")
    if not u.endswith("sitemap.xml"):
        u = f"{u}/sitemap.xml"
    if not u.startswith(("http://", "https://")):
        u = f"https://{u}"
    return u

# JavaScript snippet to click on all clickable elements (if needed)
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

def get_run_config(with_js: bool = False) -> CrawlerRunConfig:
    kwargs = {
        "cache_mode": CacheMode.BYPASS,
        "stream": False,
        "exclude_external_links": False
    }
    if with_js:
        kwargs["js_code"] = [js_click_all]
    return CrawlerRunConfig(**kwargs)

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    sp = ("You are an AI that extracts titles and summaries from web content chunks.\n"
          "Return a JSON object with 'title' and 'summary' keys.")
    uc = f"URL: {url}\n\nContent (first 1000 chars):\n{chunk[:1000]}..."
    try:
        r = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sp},
                {"role": "user", "content": uc},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        w = r.choices[0].message.content.strip()
        if w.startswith("{"):
            return json.loads(w)
        else:
            return {"title": "Untitled", "summary": w[:200]}
    except Exception as e:
        print(f"Title/Summary error: {e}")
        return {"title": "Untitled", "summary": ""}

async def process_chunk(chunk: str, num: int, url: str) -> Dict[str, Any]:
    a = await get_title_and_summary(chunk, url)
    e = await get_embedding(chunk)
    return {
        "id": f"{url}_{num}",
        "url": url,
        "chunk_number": num,
        "title": a["title"],
        "summary": a["summary"],
        "content": chunk,
        "metadata": {
            "source": urlparse(url).netloc,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
        },
        "embedding": e,
    }

async def insert_chunk_to_supabase(d: Dict[str, Any]):
    try:
        r = supabase.table("rag_chunks").upsert(d).execute()
        if r.error:
            print(r.error)
    except Exception as e:
        print(f"Insert error: {e}")

async def process_and_store_document(url: str, md: str):
    chunks = chunk_text(md)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk_to_supabase(item) for item in processed_chunks]
    await asyncio.gather(*insert_tasks)

# Recursive crawl function to follow all sitemap/internal links
async def recursive_crawl(url: str, max_depth: int = 9, current_depth: int = 0, processed: set = None):
    if processed is None:
        processed = set()
    if current_depth > max_depth or url in processed:
        return
    processed.add(url)
    
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
    run_conf = get_run_config(with_js=True)
    
    async with AsyncWebCrawler(config=bc) as crawler:
        result = await crawler.arun(url=url, config=run_conf)
        if result.success:
            print(f"Crawled: {url} (depth {current_depth})")
            # Use the passed URL for storage.
            await process_and_store_document(url, result.markdown_v2.raw_markdown)
            # Get internal links from result.links dictionary.
            links_dict = getattr(result, "links", {})
            internal_links = links_dict.get("internal", [])
            for link in internal_links:
                href = link.get("href")
                if href and same_domain(href, url) and href not in processed:
                    await recursive_crawl(href, max_depth, current_depth + 1, processed)
        else:
            print(f"Error crawling {url}: {result.error_message}")

# --- End Helper Functions ---

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

# Main Streamlit app
async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System (Supabase)", page_icon="🤖", layout="wide")
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

    ic, cc = st.columns([1, 2])
    with ic:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process.")
        url_input = st.text_input("Website URL", key="url_input", placeholder="example.com or https://example.com")
        if url_input:
            pv = format_sitemap_url(url_input)
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
                    fu = format_sitemap_url(url_input)
                    found = get_urls_from_sitemap(fu)
                    if found:
                        # Crawl every URL from the sitemap
                        for link in found:
                            await recursive_crawl(link, max_depth=9)
                    else:
                        su = url_input.rstrip("/sitemap.xml")
                        await recursive_crawl(su, max_depth=9)
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
                    st.markdown(m["content"])
            user_query = st.chat_input("Ask a question about the processed content...")
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
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
                        st.markdown(a)
                        with st.expander("References"):
                            st.markdown(dr)
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"Error: {e}")
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
