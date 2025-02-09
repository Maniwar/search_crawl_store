import os
import asyncio
import json
import requests
import streamlit as st
from datetime import datetime, timezone
from typing import List, Dict, Any
from urllib.parse import urlparse
from xml.etree import ElementTree
from dataclasses import dataclass
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in your environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def get_embedding(text: str) -> List[float]:
    try:
        r = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return r.data[0].embedding
    except:
        return [0.0]*1536

def retrieve_relevant_documentation(query: str) -> str:
    e = asyncio.run(get_embedding(query))
    r = supabase.rpc("match_documents", {"query_embedding": e, "match_count": 5}).execute()
    d = r.data
    if not d:
        return "No relevant documentation found."
    c = []
    for row in d:
        c.append(f"\n# {row['title']}\n\n{row['content'][:1000]}\n...\nSource: {row['url']}\nSimilarity: {row['similarity']:.3f}\n")
    return "\n\n---\n\n".join(c)

def list_documentation_pages() -> List[str]:
    q = "SELECT DISTINCT url FROM rag_chunks"
    r = supabase.rpc("exec", {"sql": q}).execute()
    if not r.data:
        return []
    return [row["url"] for row in r.data]

def get_page_content(u: str) -> str:
    x = supabase.table("rag_chunks").select("*").eq("url", u).execute()
    d = x.data
    if not d:
        return f"No content found for URL: {u}"
    s = sorted(d, key=lambda z: z.get("chunk_number", 0))
    t = s[0].get("title", "Untitled")
    v = [f"# {t}\n"]
    for r in s:
        v.append(r["content"])
    return "\n\n".join(v)

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
        if cb != -1 and cb > c*0.3:
            e = s + cb
        elif "\n\n" in sub:
            lb = sub.rfind("\n\n")
            if lb > c*0.3:
                e = s + lb
        elif ". " in sub:
            lp = sub.rfind(". ")
            if lp > c*0.3:
                e = s + lp + 1
        sub = t[s:e].strip()
        if sub:
            r.append(sub)
        s = max(s+1,e)
    return r

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    sp = "You are an AI that extracts titles and summaries from web content chunks.\nReturn a JSON object with 'title' and 'summary' keys."
    uc = f"URL: {url}\n\nContent (first 1000 chars):\n{chunk[:1000]}..."
    try:
        r = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4"),
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
    except:
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
    except:
        pass

async def process_and_store_document(url: str, md: str):
    c = chunk_text(md)
    t = []
    for i, ct in enumerate(c):
        t.append(process_chunk(ct, i, url))
    p = await asyncio.gather(*t)
    s = []
    for x in p:
        s.append(insert_chunk_to_supabase(x))
    await asyncio.gather(*s)

def get_urls_from_sitemap(u: str) -> List[str]:
    try:
        r = requests.get(u)
        r.raise_for_status()
        ro = ElementTree.fromstring(r.content)
        ns = {"ns":"http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text for loc in ro.findall(".//ns:loc", ns)]
    except:
        return []

async def crawl_parallel(urls: List[str], mc: int = 5):
    bc = BrowserConfig(headless=True, verbose=False, extra_args=["--disable-gpu","--disable-dev-shm-usage","--no-sandbox"])
    cc = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    c = AsyncWebCrawler(config=bc)
    await c.start()
    try:
        sem = asyncio.Semaphore(mc)
        t = len(urls)
        d = 0
        async def run_url(u: str):
            nonlocal d
            async with sem:
                r = await c.arun(url=u, config=cc, session_id="session1")
                if r.success:
                    d += 1
                    await process_and_store_document(u, r.markdown_v2.raw_markdown)
        await asyncio.gather(*(run_url(u) for u in urls))
    finally:
        await c.close()

def format_sitemap_url(u: str) -> str:
    u = u.rstrip("/")
    if not u.endswith("sitemap.xml"):
        u = f"{u}/sitemap.xml"
    if not u.startswith(("http://","https://")):
        u = f"https://{u}"
    return u

def delete_all_chunks():
    supabase.table("rag_chunks").delete().neq("id", "").execute()

def get_db_stats():
    try:
        r = supabase.table("rag_chunks").select("id, url, metadata").execute()
        d = r.data
        if not d:
            return {"urls":[],"domains":[],"doc_count":0,"last_updated":None}
        u = set(x["url"] for x in d)
        dm = set(x["metadata"].get("source","") for x in d)
        c = len(d)
        lt = [x["metadata"].get("crawled_at",None) for x in d]
        lt = [z for z in lt if z]
        if not lt:
            return {"urls":list(u),"domains":list(dm),"doc_count":c,"last_updated":None}
        mx = max(lt)
        dt = datetime.fromisoformat(mx.replace("Z","+00:00"))
        tz = datetime.now().astimezone().tzinfo
        dt = dt.astimezone(tz)
        lu = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        return {"urls":list(u),"domains":list(dm),"doc_count":c,"last_updated":lu}
    except:
        return None

async def main():
    st.set_page_config(page_title="Dynamic RAG Chat System (Supabase)",page_icon="🤖",layout="wide")
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete=False
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed=set()
    if "is_processing" not in st.session_state:
        st.session_state.is_processing=False
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions=None
    st.title("Dynamic RAG Chat System (Supabase)")
    s=get_db_stats()
    if s and s["doc_count"]>0:
        st.session_state.processing_complete=True
        st.session_state.urls_processed=set(s["urls"])
    if s and s["doc_count"]>0:
        st.success("💡 System is ready with existing knowledge base (Supabase)!")
        with st.expander("Knowledge Base Information",expanded=True):
            st.markdown(f"""\n### Current Knowledge Base Stats:\n- **Documents**: {s['doc_count']}\n- **Sources**: {len(s['domains'])}\n- **Last updated**: {s['last_updated']}\n\n### Sources include:\n{', '.join(s['domains'])}\n\n### You can ask questions about:\n- Any content from the processed websites\n- Specific information from any of the loaded pages\n- Technical details, documentation, or other content from these sources\n\n### Loaded URLs:\n""")
            for uu in s["urls"]:
                st.write(f"- {uu}")
    else:
        st.info("👋 Welcome! Start by adding a website to create your knowledge base.")
    ic, cc = st.columns([1,2])
    with ic:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process.")
        url_input = st.text_input("Website URL",key="url_input",placeholder="example.com or https://example.com")
        if url_input:
            pv=format_sitemap_url(url_input)
            st.caption(f"Will try: {pv}")
        c1,c2=st.columns(2)
        with c1:
            pb=st.button("Process URL",disabled=st.session_state.is_processing)
        with c2:
            if st.button("Clear Database",disabled=st.session_state.is_processing):
                delete_all_chunks()
                st.session_state.processing_complete=False
                st.session_state.urls_processed=set()
                st.session_state.messages=[]
                st.session_state.suggested_questions=None
                st.success("Database cleared successfully!")
                st.experimental_rerun()
        if pb and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing=True
                with st.spinner("Crawling & Processing..."):
                    fu=format_sitemap_url(url_input)
                    found=get_urls_from_sitemap(fu)
                    if found:
                        asyncio.run(crawl_parallel(found))
                    else:
                        su=url_input.rstrip("/sitemap.xml")
                        asyncio.run(crawl_parallel([su]))
                st.session_state.urls_processed.add(url_input)
                st.session_state.processing_complete=True
                st.session_state.is_processing=False
                st.experimental_rerun()
            else:
                st.warning("This URL has already been processed!")
        if st.session_state.urls_processed:
            st.subheader("Processed URLs:")
            up=list(st.session_state.urls_processed)
            for x in up[:3]:
                st.write(f"✓ {x}")
            r=len(up)-3
            if r>0:
                st.write(f"_...and {r} more_")
                with st.expander("Show all URLs"):
                    for uxx in up[3:]:
                        st.write(f"✓ {uxx}")
    with cc:
        if st.session_state.processing_complete:
            st.subheader("Chat Interface")
            for m in st.session_state.messages:
                if m.get("role")=="user":
                    with st.chat_message("user"):
                        st.markdown(m["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(m["content"])
            user_query=st.chat_input("Ask a question about the processed content...")
            if user_query:
                st.session_state.messages.append({"role":"user","content":user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
                dr=retrieve_relevant_documentation(user_query)
                sys="You have access to the following context:\n"+dr+"\nAnswer the question."
                msgs=[
                    {"role":"system","content":sys},
                    {"role":"user","content":user_query}
                ]
                try:
                    r=asyncio.run(openai_client.chat.completions.create(model=os.getenv("LLM_MODEL","gpt-4"),messages=msgs))
                    a=r.choices[0].message.content
                    st.session_state.messages.append({"role":"assistant","content":a})
                    with st.chat_message("assistant"):
                        st.markdown(a)
                except Exception as e:
                    st.session_state.messages.append({"role":"assistant","content":f"Error: {e}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"Error: {e}")
            if st.button("Clear Chat History",type="secondary"):
                st.session_state.messages=[]
                st.experimental_rerun()
        else:
            st.info("Please process a URL first to start chatting!")
    st.markdown("---")
    if s and s["doc_count"]>0:
        st.markdown(f"System Status: 🟢 Ready with {s['doc_count']} documents from {len(s['domains'])} sources")
    else:
        st.markdown("System Status: 🟡 Waiting for content")

if __name__=="__main__":
    asyncio.run(main())
