# Streamlit + Supabase RAG with pydantic_ai + crawl4ai
# Convert the user's existing code to use Supabase as the vector DB instead of Chroma
# This code is a single Streamlit app that can be deployed on Streamlit Cloud.
# Make sure to configure the following environment variables in your Streamlit secrets:
#   - SUPABASE_URL
#   - SUPABASE_SERVICE_ROLE_KEY
#   - OPENAI_API_KEY
#
# A table named "rag_chunks" in Supabase is assumed, with the following columns:
#   - id: text (PRIMARY KEY)
#   - url: text
#   - chunk_number: integer
#   - title: text
#   - summary: text
#   - content: text
#   - metadata: jsonb
#   - embedding: vector(1536)
#
# The Postgres "pgvector" extension is required for vector similarity.
# Additionally, you must define or create an RPC function "match_documents" in Supabase:
#   create or replace function match_documents(
#       query_embedding vector(1536),
#       match_count int default 5
#   )
#   returns table (
#       id text,
#       url text,
#       chunk_number int,
#       title text,
#       summary text,
#       content text,
#       metadata jsonb,
#       similarity float
#   )
#   language sql stable
#   as $$
#       select
#           id,
#           url,
#           chunk_number,
#           title,
#           summary,
#           content,
#           metadata,
#           1 - (rag_chunks.embedding <-> query_embedding) as similarity
#       from rag_chunks
#       order by rag_chunks.embedding <-> query_embedding
#       limit match_count;
#   $$;


import os
import asyncio
import json
import requests
import streamlit as st

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Literal
from urllib.parse import urlparse
from xml.etree import ElementTree

# Additional imports from user code
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)
from pydantic_ai_agent import pydantic_ai_agent, PydanticAIDeps
from supabase import create_client, Client
from openai import AsyncOpenAI
import logfire
from dotenv import load_dotenv

# crawl4ai imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

########################
# Load environment variables
########################

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise ValueError("Please set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and OPENAI_API_KEY in your environment.")

########################
# Supabase + OpenAI setup
########################
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

llm = os.getenv("LLM_MODEL", "gpt-4")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="never")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

########################
# pydantic_ai setup
########################

@dataclass
class MyPydanticAIDeps:
    # We can store references to supabase, openai, etc.
    supabase: Client
    openai_client: AsyncOpenAI


system_prompt = """
You are an expert assistant with access to a knowledge base of documentation and content.
Your job is to help users understand and work with the content they've provided.

Always make sure you look at the relevant documentation before answering unless you're certain about the answer.
Be honest when you can't find relevant information in the knowledge base.

When analyzing the content, make sure to:
1. Provide accurate information based on the stored content
2. Cite specific examples when possible
3. Be clear when you're making assumptions or inferring information
"""

pydantic_ai_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MyPydanticAIDeps,
    retries=2
)

########################
# Utility: vector embedding with OpenAI
########################

async def get_embedding(text: str) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero vector if error
        return [0.0]*1536

########################
# Tools for the pydantic_ai_agent
########################

@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[MyPydanticAIDeps], user_query: str) -> str:
    """Retrieve relevant documentation chunks from Supabase based on the query."""
    try:
        query_embedding = await get_embedding(user_query)
        match_count = 5
        rpc_res = ctx.deps.supabase.rpc(
            "match_documents",
            {"query_embedding": query_embedding, "match_count": match_count}
        ).execute()

        data = rpc_res.data
        if not data:
            return "No relevant documentation found."

        formatted_chunks = []
        for row in data:
            chunk_text = f"""
# {row['title']}

{row['content'][:1000]}\n...\n
Source: {row['url']}\nSimilarity: {row['similarity']:.3f}
"""
            formatted_chunks.append(chunk_text)
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@pydantic_ai_agent.tool
async def list_documentation_pages(ctx: RunContext[MyPydanticAIDeps]) -> List[str]:
    """Retrieve a list of all URLs from the rag_chunks table."""
    try:
        # We'll do a distinct query on url
        query = """
        SELECT DISTINCT url
        FROM rag_chunks
        """
        res = ctx.deps.supabase.rpc("exec", {"sql": query}).execute()
        # If you haven't set up a custom rpc for direct sql, you can create a function or do a supabase client
        if not res.data:
            return []
        return [row["url"] for row in res.data]
    except Exception as e:
        print(f"Error listing documentation pages: {e}")
        return []


@pydantic_ai_agent.tool
async def get_page_content(ctx: RunContext[MyPydanticAIDeps], url: str) -> str:
    """Retrieve the full combined content of a specific URL from the rag_chunks table."""
    try:
        # fetch all chunks for a given url
        res = ctx.deps.supabase.table("rag_chunks") \
            .select("*") \
            .eq("url", url) \
            .execute()
        data = res.data
        if not data:
            return f"No content found for URL: {url}"
        # sort by chunk_number
        data_sorted = sorted(data, key=lambda x: x.get("chunk_number", 0))
        # combine all content
        combined = []
        page_title = data_sorted[0].get("title", "Untitled")
        combined.append(f"# {page_title}\n")
        for row in data_sorted:
            combined.append(row["content"])
        return "\n\n".join(combined)
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

########################
# Web crawler and chunk insertion
########################

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4 or given model."""
    system_prompt = """You are an AI that extracts titles and summaries from web content chunks.\nReturn a JSON object with 'title' and 'summary' keys."""
    try:
        # We'll do a best-effort prompt.
        # You could also do a structured function call if your model supports it.
        user_content = f"URL: {url}\n\nContent (first 1000 chars):\n{chunk[:1000]}..."
        resp = await openai_client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        # naive parse for JSON if present
        # If not present, fallback
        if raw.startswith("{"):
            return json.loads(raw)
        else:
            # fallback
            return {"title": "Untitled", "summary": raw[:200]}
    except Exception as e:
        print(f"Error getting title/summary: {e}")
        return {"title": "Untitled", "summary": ""}

async def process_chunk(chunk: str, chunk_number: int, url: str) -> Dict[str, Any]:
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    return {
        "id": f"{url}_{chunk_number}",
        "url": url,
        "chunk_number": chunk_number,
        "title": extracted["title"],
        "summary": extracted["summary"],
        "content": chunk,
        "metadata": {
            "source": urlparse(url).netloc,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
        },
        "embedding": embedding,
    }

async def insert_chunk_to_supabase(chunk_data: Dict[str, Any]):
    try:
        # Insert or upsert row into rag_chunks
        res = supabase.table("rag_chunks").upsert(chunk_data).execute()
        if res.error:
            print("Error inserting chunk:", res.error)
        else:
            print(f"Inserted chunk {chunk_data['id']} for {chunk_data['url']}")
    except Exception as e:
        print(f"Error inserting chunk: {e}")

async def process_and_store_document(url: str, markdown: str):
    chunks = chunk_text(markdown)
    tasks = []
    for i, chunk_text_val in enumerate(chunks):
        tasks.append(process_chunk(chunk_text_val, i, url))
    processed = await asyncio.gather(*tasks)
    insert_tasks = []
    for item in processed:
        insert_tasks.append(insert_chunk_to_supabase(item))
    await asyncio.gather(*insert_tasks)

########################
# Crawler for sitemaps or single URLs
########################

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    print(f"[crawl_parallel] Found {len(urls)} URLs to crawl.")
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        total_urls = len(urls)
        processed_urls = 0

        async def process_url(url: str):
            nonlocal processed_urls
            async with semaphore:
                result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
                if result.success:
                    processed_urls += 1
                    print(f"Successfully crawled: {url} ({processed_urls}/{total_urls})")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        await asyncio.gather(*(process_url(u) for u in urls))
        print(f"Completed crawling {processed_urls} / {total_urls} URLs")
    finally:
        await crawler.close()

########################
# Streamlit app
########################

def format_sitemap_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("sitemap.xml"):
        url = f"{url}/sitemap.xml"
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def delete_all_chunks():
    # caution: remove all data from rag_chunks
    supabase.table("rag_chunks").delete().neq("id", "").execute()


def get_db_stats():
    """Get stats from Supabase rag_chunks table."""
    try:
        # fetch all rows
        res = supabase.table("rag_chunks").select("id, url, metadata").execute()
        data = res.data
        if not data:
            return {
                "urls": [],
                "domains": [],
                "doc_count": 0,
                "last_updated": None,
            }
        urls = set(row["url"] for row in data)
        domains = set(row["metadata"].get("source", "") for row in data)
        doc_count = len(data)
        # find max crawled_at
        last_updated_times = [row["metadata"].get("crawled_at", None) for row in data]
        last_updated_times = [x for x in last_updated_times if x]
        if not last_updated_times:
            last_updated = None
        else:
            last_updated_iso = max(last_updated_times)
            dt = datetime.fromisoformat(last_updated_iso.replace("Z", "+00:00"))
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.astimezone(local_tz)
            last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        return {
            "urls": list(urls),
            "domains": list(domains),
            "doc_count": doc_count,
            "last_updated": last_updated,
        }
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return None


async def main():
    st.set_page_config(
        page_title="Dynamic RAG Chat System (Supabase)",
        page_icon="🤖",
        layout="wide"
    )

    # Session state init
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

    # Check existing data
    stats = get_db_stats()
    if stats and stats["doc_count"] > 0:
        st.session_state.processing_complete = True
        st.session_state.urls_processed = set(stats["urls"])

    if stats and stats["doc_count"] > 0:
        st.success("💡 System is ready with existing knowledge base (Supabase)!")
        with st.expander("Knowledge Base Information", expanded=True):
            st.markdown(
                f"""
### Current Knowledge Base Stats:
- **Documents**: {stats['doc_count']}
- **Sources**: {len(stats['domains'])}
- **Last updated**: {stats['last_updated']}

### Sources include:
{', '.join(stats['domains'])}

### You can ask questions about:
- Any content from the processed websites
- Specific information from any of the loaded pages
- Technical details, documentation, or other content from these sources

### Loaded URLs:
"""
            )
            for u in stats["urls"]:
                st.write(f"- {u}")
    else:
        st.info("👋 Welcome! Start by adding a website to create your knowledge base.")

    input_col, chat_col = st.columns([1, 2])

    with input_col:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process. The system will:")
        st.write("1. Try to find and process the sitemap (by appending '/sitemap.xml')")
        st.write("2. If no sitemap is found, process the URL as a single page")

        url_input = st.text_input(
            "Website URL",
            key="url_input",
            placeholder="example.com or https://example.com",
        )

        if url_input:
            formatted_preview = format_sitemap_url(url_input)
            st.caption(f"Will try: {formatted_preview}")

        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("Process URL", disabled=st.session_state.is_processing)
        with col2:
            if st.button("Clear Database", disabled=st.session_state.is_processing):
                delete_all_chunks()
                st.session_state.processing_complete = False
                st.session_state.urls_processed = set()
                st.session_state.messages = []
                st.session_state.suggested_questions = None
                st.success("Database cleared successfully!")
                st.experimental_rerun()

        if process_button and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing = True
                with st.spinner("Crawling & Processing..."):
                    formatted_url = format_sitemap_url(url_input)
                    urls = get_urls_from_sitemap(formatted_url)
                    if urls:
                        await crawl_parallel(urls)
                    else:
                        # single URL
                        # remove /sitemap.xml if present
                        single_url = url_input.rstrip("/sitemap.xml")
                        await crawl_parallel([single_url])
                st.session_state.urls_processed.add(url_input)
                st.session_state.processing_complete = True
                st.session_state.is_processing = False
                st.experimental_rerun()
            else:
                st.warning("This URL has already been processed!")

        # Display processed URLs
        if st.session_state.urls_processed:
            st.subheader("Processed URLs:")
            up_list = list(st.session_state.urls_processed)
            for x in up_list[:3]:
                st.write(f"✓ {x}")
            remaining = len(up_list) - 3
            if remaining > 0:
                st.write(f"_...and {remaining} more_")
                with st.expander("Show all URLs"):
                    for urlx in up_list[3:]:
                        st.write(f"✓ {urlx}")

    with chat_col:
        if st.session_state.processing_complete:
            st.subheader("Chat Interface")
            # Render chat messages
            for msg in st.session_state.messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, UserPromptPart):
                            with st.chat_message("user"):
                                st.markdown(part.content)
                elif isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        if isinstance(part, TextPart):
                            with st.chat_message("assistant"):
                                st.markdown(part.content)

            user_query = st.chat_input("Ask a question about the processed content...")
            if user_query:
                st.session_state.messages.append(
                    ModelRequest(parts=[UserPromptPart(content=user_query)])
                )
                with st.chat_message("user"):
                    st.markdown(user_query)

                # run agent with streaming
                async with pydantic_ai_agent.run_stream(
                    user_query,
                    deps=MyPydanticAIDeps(
                        supabase=supabase,
                        openai_client=openai_client,
                    ),
                    message_history=st.session_state.messages[:-1],
                ) as result:
                    partial_text = ""
                    msg_placeholder = st.empty()
                    async for chunk in result.stream_text(delta=True):
                        partial_text += chunk
                        msg_placeholder.markdown(partial_text)
                    st.session_state.messages.extend(result.new_messages())

            if st.button("Clear Chat History", type="secondary"):
                st.session_state.messages = []
                st.experimental_rerun()
        else:
            st.info("Please process a URL first to start chatting!")

    # Footer
    st.markdown("---")
    if stats and stats["doc_count"] > 0:
        st.markdown(
            f"System Status: 🟢 Ready with {stats['doc_count']} documents from {len(stats['domains'])} sources"
        )
    else:
        st.markdown("System Status: 🟡 Waiting for content")

if __name__ == "__main__":
    asyncio.run(main())
