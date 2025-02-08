import re
import asyncio
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import nest_asyncio
import faiss
from urllib.parse import urlencode
from supabase import create_client, Client

# Import Crawl4AI classes including BrowserConfig and CrawlerRunConfig
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Import torch and CLIP modules for image embeddings
import torch
from transformers import CLIPProcessor, CLIPModel

# Import the new OpenAI client
from openai import OpenAI

# === IMPORTANT ===
# On Streamlit Cloud (or locally), ensure that Playwright browsers are installed.
# You can do this by adding a post-install command: "playwright install"
# or by running "playwright install" manually.

# Enable nested event loops (required for async code in Streamlit)
nest_asyncio.apply()

# Initialize OpenAI client with the new API client method.
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Supabase client.
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Configure the Streamlit page.
st.set_page_config(page_title="Local Listings Shopping Session", layout="wide")
st.title("Local Listings Shopping Session")
st.write("Find authentic listings from Facebook Marketplace or Craigslist in your area—and let AI guide your shopping session!")

# -- User Inputs --
source = st.selectbox("Select Source", options=["Facebook Marketplace", "Craigslist"])
zip_code = st.text_input("Enter your Zip Code (5 digits):", value="60614")
keyword = st.text_input("Optional: Enter a product keyword (e.g., 'bike', 'sofa'):")

# Build a target URL based on the selected source.
def construct_target_url(source: str, zip_code: str, keyword: str) -> str:
    if source == "Craigslist":
        params = {"postal": zip_code}
        if keyword:
            params["query"] = keyword
        # Example: using a generic Craigslist domain (adjust as needed)
        return f"https://sfbay.craigslist.org/search/sss?{urlencode(params)}"
    elif source == "Facebook Marketplace":
        params = {"postal": zip_code}
        if keyword:
            params["query"] = keyword
        return f"https://www.facebook.com/marketplace/learnmore?{urlencode(params)}"
    else:
        return ""

target_url = construct_target_url(source, zip_code, keyword)
st.write("Target URL:", target_url)

# --- Advanced Crawl4AI Setup ---
# Define a BrowserConfig for global browser settings.
browser_cfg = BrowserConfig(
    browser_type="chromium",
    headless=True,
    verbose=True
)

# We'll use a CrawlerRunConfig for this crawl.
crawl_config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS
)

# --- Asynchronous Functions for Scraping and Filtering Listings ---

async def crawl_listings(url: str) -> str:
    # Use AsyncWebCrawler with our BrowserConfig.
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        # Assume the result returns a Markdown string in result.markdown_v2.fit_markdown.
        return result.markdown_v2.fit_markdown

async def is_listing_real(listing_text: str) -> bool:
    prompt = (
        "You are an expert in detecting fraudulent online listings. "
        "Determine whether the following listing is authentic (real) or fake/spam. "
        "Respond with a single word: 'real' or 'fake'.\n\n"
        f"Listing:\n{listing_text}\n\nAnswer:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a listing authenticity classifier. Answer only with 'real' or 'fake'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer == "real"
    except Exception as e:
        st.error(f"Classification error: {e}")
        return False

def extract_zip(text: str) -> str:
    match = re.search(r"\b(\d{5})\b", text)
    return match.group(1) if match else ""

async def process_url(url: str, desired_zip: str, keyword: str) -> list:
    page_text = await crawl_listings(url)
    # Split the scraped text into candidate listings (assume listings separated by two newlines)
    candidate_listings = [item.strip() for item in page_text.split("\n\n") if item.strip()]
    filtered_listings = []
    for listing in candidate_listings:
        if extract_zip(listing) != desired_zip:
            continue
        if keyword and keyword.lower() not in listing.lower():
            continue
        if await is_listing_real(listing):
            filtered_listings.append(listing)
    return filtered_listings

# --- Functions for Computing Embeddings ---

def get_text_embeddings(texts: list) -> np.ndarray:
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        embeddings = [record["embedding"] for record in response["data"]]
        return np.array(embeddings).astype("float32")
    except Exception as e:
        st.error(f"Error computing text embeddings: {e}")
        return np.empty((0, 1536), dtype="float32")

# Load CLIP model and processor for image embeddings.
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def get_image_embedding(image_url: str) -> np.ndarray:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding.cpu().numpy().astype("float32")[0]
    except Exception as e:
        st.error(f"Error processing image from {image_url}: {e}")
        return None

# --- Supabase Integration: Store Listing Metadata ---
# Note: The exec_sql RPC may not exist. In production, create the table manually via Supabase.
def ensure_listings_table():
    query = """
    CREATE TABLE IF NOT EXISTS listings (
      id SERIAL PRIMARY KEY,
      source TEXT,
      zip_code TEXT,
      keyword TEXT,
      listing_text TEXT,
      image_url TEXT,
      text_embedding JSONB,
      image_embedding JSONB,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        # Attempt to run the query via an RPC.
        # If the function doesn't exist, catch and warn.
        supabase.rpc("exec_sql", {"sql": query}).execute()
    except Exception as e:
        st.warning("Could not run exec_sql; please ensure the 'listings' table exists manually in Supabase.")

ensure_listings_table()

def store_listing_metadata(listing: dict):
    try:
        response = supabase.table("listings").insert(listing).execute()
        if response.get("error"):
            st.error(f"Supabase insert error: {response['error']['message']}")
    except Exception as e:
        st.error(f"Error storing listing in Supabase: {e}")

# --- Build FAISS Index ---
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_faiss(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 3):
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# --- UI: Render a Listing Card ---
def render_listing_card(listing: dict):
    text = listing.get("listing_text", "")
    title = text[:60] + ("..." if len(text) > 60 else "")
    zip_found = listing.get("zip_code", "")
    card_html = f"""
    <div style="
         border: 1px solid #ccc;
         border-radius: 8px;
         padding: 16px;
         margin: 8px;
         box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
         background-color: #fff;">
      <h4 style="margin-bottom: 4px;">{title}</h4>
      <p style="font-size: 14px; color: #555;">Zip Code: {zip_found}</p>
      <p style="font-size: 13px; color: #333;">{text[:200]}{"..." if len(text) > 200 else ""}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- Main Execution ---
if st.button("Search Listings"):
    if not target_url or not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
        st.error("Please enter a valid target URL and a 5-digit zip code.")
    else:
        st.info("Scraping and filtering listings. Please wait...")
        try:
            real_listings = asyncio.run(process_url(target_url, zip_code, keyword))
            st.success(f"Found {len(real_listings)} authentic listings matching zip code {zip_code}!")
            
            # Enrich listings with embeddings and optionally extract an image URL.
            image_url_pattern = re.compile(r'(https?://\S+\.(jpg|jpeg|png))', re.IGNORECASE)
            enriched_listings = []
            for listing in real_listings:
                image_url = None
                match = image_url_pattern.search(listing)
                if match:
                    image_url = match.group(0)
                listing_data = {
                    "source": source,
                    "zip_code": zip_code,
                    "keyword": keyword,
                    "listing_text": listing,
                    "image_url": image_url,
                }
                # Compute text embedding.
                text_emb = get_text_embeddings([listing])
                listing_data["text_embedding"] = text_emb.tolist()[0]
                # Optionally compute image embedding.
                if image_url:
                    img_emb = get_image_embedding(image_url)
                    listing_data["image_embedding"] = img_emb.tolist() if img_emb is not None else None
                else:
                    listing_data["image_embedding"] = None
                enriched_listings.append(listing_data)
                store_listing_metadata(listing_data)
            
            # Build FAISS index for text embeddings for similarity search.
            text_embeddings = np.array([np.array(d["text_embedding"], dtype="float32") for d in enriched_listings])
            text_index = build_faiss_index(text_embeddings)
            
            st.session_state["listings"] = enriched_listings
            st.session_state["text_faiss_index"] = text_index

            # Display listings as cards in a grid layout.
            num_columns = 3
            cols = st.columns(num_columns)
            for idx, listing in enumerate(enriched_listings):
                with cols[idx % num_columns]:
                    render_listing_card(listing)
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# --- Shopping Session: Similarity Search Within Stored Listings ---
st.markdown("### Shopping Session: Find Similar Listings by Text")
query_text = st.text_input("Enter text to search for similar listings within this session:")
if query_text and "text_faiss_index" in st.session_state:
    try:
        query_emb = get_text_embeddings([query_text])
        query_emb = np.array(query_emb, dtype="float32").reshape(1, -1)
        index: faiss.IndexFlatL2 = st.session_state["text_faiss_index"]
        if index is None or index.ntotal == 0:
            st.warning("No listings available for similarity search.")
        else:
            distances, indices = search_faiss(index, query_emb, k=3)
            st.markdown("**Similar Listings:**")
            for i in indices[0]:
                if i < len(st.session_state["listings"]):
                    render_listing_card(st.session_state["listings"][i])
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
