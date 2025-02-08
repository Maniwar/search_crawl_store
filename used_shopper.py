# app.py
import os
import subprocess
import re
import asyncio
import json
from io import BytesIO
from urllib.parse import urlencode

import streamlit as st
import numpy as np
import requests
from PIL import Image
import nest_asyncio
import faiss
# Disable Streamlit file watcher to suppress torch __path__ errors
os.environ["STREAMLIT_WATCHER_DISABLED"] = "true"
import torch
from supabase import create_client, Client
from transformers import CLIPProcessor, CLIPModel
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from openai import OpenAI

# Enable nested event loops for async code to avoid errors
nest_asyncio.apply()

# st.set_page_config MUST be the very first Streamlit command.
st.set_page_config(page_title="Local Listings Shopping Session", layout="wide")

# --- Attempt to run the Playwright browser install script ---
if os.path.exists("install_browsers.sh"):
  try:
    subprocess.run(["bash", "install_browsers.sh"], check=True)
    st.info("Playwright browsers installation script executed.")
  except subprocess.CalledProcessError as e:
    st.warning(f"Playwright installation script failed: {e}")
  except Exception as e:
    st.warning(f"Unexpected error during Playwright installation script: {e}")
else:
  st.info("install_browsers.sh not found; skipping browser installation script.")

# --- Function to check Node.js version and Playwright browsers ---
def check_playwright_browsers():
  node_version_str = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
  node_major_version = int(node_version_str.split('.')[0].replace('v', ''))

  if node_major_version < 14:
    st.error(f"Your Node.js version is {node_version_str}. Playwright requires Node.js 14 or higher. Please update your Node.js version to use Playwright.")
    return False

  try:
    subprocess.run(["which", "playwright"], check=True, capture_output=True)
    st.info("Playwright command found in PATH.")
    return True
  except subprocess.CalledProcessError:
    st.warning("Playwright command not found in PATH. Attempting to install browsers...")
    try:
      subprocess.run(["npx", "playwright", "install", "chromium"], check=True)
      st.success("Playwright browsers installed successfully via npx playwright install chromium.")
      return True
    except Exception as e:
      st.error(f"Failed to install Playwright browsers automatically: {e}. Please ensure Playwright is installed and in PATH.")
      st.error("You may need to run 'npx playwright install chromium' manually in the shell if automatic installation fails.")
      return False
  except Exception as e:
    st.error(f"Error checking Playwright installation: {e}")
    return False

if not check_playwright_browsers():
  st.stop() # Stop the app if browsers are not installed or Node.js version is too old

# Initialize API Clients
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

st.title("Local Listings Shopping Session")
st.write("Find authentic listings from Facebook Marketplace and Craigslist in your area—and let AI guide your shopping session!")

# --- User Inputs ---
selected_sources = st.multiselect(
  "Select Sources",
  options=["Facebook Marketplace", "Craigslist"],
  default=["Craigslist", "Facebook Marketplace"]
)
zip_code = st.text_input("Enter your Zip Code (5 digits):", value="60614")
search_description = st.text_input("Describe what you're looking for (e.g. 'affordable sports car'):")

# --- Dynamic Search Parameter Generation ---
@st.cache_data
def generate_search_parameters_dynamic(description: str) -> dict:
  prompt = (
    "You are a dynamic search assistant. Given a free-form product search description, output a JSON object with exactly "
    "two keys: \"refined_query\" and \"filters\". The \"refined_query\" should be a concise phrase for the core search term. "
    "The \"filters\" should be an object containing additional filter criteria such as make, model, year_range, trim, price_range, "
    "or any other parameters relevant to the product. If no extra filters apply, output an empty object for filters.\n\n"
    "Example:\n"
    "Input: \"affordable sports car\"\n"
    "Output: {\"refined_query\": \"affordable sports car\", \"filters\": {\"make\": \"Mazda\", \"model\": \"MX-5 Miata\", \"year_range\": \"2000-2010\", \"trim\": \"Base\", \"price_range\": \"5000-15000\"}}\n\n"
    "Input: \"modern minimalist dining table\"\n"
    "Output: {\"refined_query\": \"minimalist dining table\", \"filters\": {\"style\": \"modern\", \"material\": \"wood\"}}\n\n"
    f"Now, given the following description, output only a valid JSON object with these two keys:\n{description}\n\nOutput JSON:"
  )
  try:
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "Output only valid JSON with keys 'refined_query' and 'filters'."},
          {"role": "user", "content": prompt}
      ],
      temperature=0.5,
      max_tokens=200
    )
    text = response.choices[0].message.content.strip()
    params = json.loads(text)
    return params
  except Exception as e:
    st.error(f"Error generating dynamic search parameters: {e}")
    return {"refined_query": description, "filters": {}}

def refine_query_from_candidate(candidate: dict) -> str:
  parts = []
  for key in ["make", "model", "year_range", "trim"]:
    value = candidate.get(key)
    if value and value != "null":
      parts.append(str(value))
  if not parts:
    parts.append(candidate.get("refined_query", ""))
  return " ".join(parts).strip()

# --- URL Construction ---
def construct_target_url_dynamic(source: str, zip_code: str, candidate: dict) -> str:
  refined_query = refine_query_from_candidate(candidate)
  filters = candidate.get("filters", {})
  if source == "Craigslist":
    params = {"postal": zip_code}
    if refined_query:
      params["query"] = refined_query
    if "price_range" in filters and filters["price_range"] not in [None, "null"]:
      try:
        min_price, max_price = filters["price_range"].split("-")
        params["min_price"] = min_price.strip()
        params["max_price"] = max_price.strip()
      except Exception:
        pass
    return f"https://sfbay.craigslist.org/search/sss?{urlencode(params)}" # Using sfbay as default craigslist zone
  elif source == "Facebook Marketplace":
    params = {"radius": 10, "zip": zip_code, "query": refined_query} # radius in miles
    return f"https://www.facebook.com/marketplace/search?{urlencode(params)}"
  else:
    return ""

# --- Asynchronous Crawling & Filtering ---
async def crawl_listings(url: str) -> str:
  run_config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    stream=True
  )
  try:
    async with AsyncWebCrawler() as crawler:
      result = await crawler.arun(url=url, config=run_config)
      return result.markdown_v2.fit_markdown
  except Exception as e:
    error_message = f"Error during crawling at URL {url}: {e}"
    st.error(error_message)
    st.error("Please ensure that Playwright browsers are installed and in your PATH. If issues persist, try running 'npx playwright install chromium' in the shell.")
    return ""

@st.cache_data
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
          {"role": "system", "content": "Answer only with 'real' or 'fake'."},
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

async def process_source(source: str, zip_code: str, candidate: dict) -> list:
  url = construct_target_url_dynamic(source, zip_code, candidate)
  if not url:
    return []
  page_text = await crawl_listings(url)
  if not page_text: # Handle empty crawl results gracefully
    return []
  candidate_listings = [item.strip() for item in page_text.split("\n\n") if item.strip()]
  filtered_listings = []
  refined_query = refine_query_from_candidate(candidate)
  for listing in candidate_listings:
    if extract_zip(listing) != zip_code and source == "Craigslist": # only check zip for craigslist
      continue # Skip if zip code doesn't match for craigslist
    if refined_query and refined_query.lower() not in listing.lower():
      continue
    if await is_listing_real(listing):
      filtered_listings.append((source, listing))
  return filtered_listings

# --- Embeddings Functions ---
@st.cache_data
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

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

@st.cache_data
def get_image_embedding(image_url: str) -> np.ndarray:
  try:
    response = requests.get(image_url, stream=True) # stream=True for better handling of large images
    if response.status_code == 200: # Check if the image was successfully fetched
      image = Image.open(BytesIO(response.content)).convert("RGB")
      inputs = clip_processor(images=image, return_tensors="pt")
      with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
      embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
      return embedding.cpu().numpy().astype("float32")[0]
    else:
      st.warning(f"Failed to fetch image from {image_url} (Status code: {response.status_code})")
      return None
  except Exception as e:
    st.error(f"Error processing image from {image_url}: {e}")
    return None

# --- Supabase Integration ---
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

# --- UI Rendering ---
def render_listing_card(listing: dict):
  text = listing.get("listing_text", "")
  title = text[:60] + ("..." if len(text) > 60 else "")
  zip_found = listing.get("zip_code", "")
  source = listing.get("source", "Unknown")
  image_url = listing.get("image_url", None)

  card_html = f"""
  <div style="
       border: 1px solid #ccc;
       border-radius: 8px;
       padding: 16px;
       margin: 8px;
       box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
       background-color: #fff;">
    <h4 style="margin-bottom: 4px;">{title}</h4>
    <p style="font-size: 14px; color: #555;">Source: {source} | Zip Code: {zip_found}</p>
    """
  if image_url:
    card_html += f'<img src="{image_url}" style="max-width:100%; height:auto; margin-bottom: 8px;">'
  card_html += f"""
    <p style="font-size: 13px; color: #333;">{text[:200]}{"..." if len(text) > 200 else ""}</p>
  </div>
  """
  st.markdown(card_html, unsafe_allow_html=True)

# --- Main Execution for "Search Listings" ---
if st.button("Search Listings"):
  if not selected_sources or not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
    st.error("Please select at least one source and enter a valid 5-digit zip code.")
  else:
    async def run_search():
      st.info("Generating dynamic search parameters...")
      dynamic_params = (generate_search_parameters_dynamic(search_description)
                        if search_description else {"refined_query": search_description, "filters": {}})
      st.write("Dynamic search parameters:", dynamic_params)

      combined_results = []
      candidates = dynamic_params if isinstance(dynamic_params, list) else [dynamic_params]
      for candidate in candidates:
        for source in selected_sources:
          st.info(f"Crawling {source} for '{candidate.get('refined_query', search_description)}'...") # User feedback
          results = await process_source(source, zip_code, candidate)
          combined_results.extend(results)

      st.success(f"Found {len(combined_results)} authentic listings matching zip code {zip_code}!")

      image_url_pattern = re.compile(r'(https?://\S+\.(jpg|jpeg|png))', re.IGNORECASE)
      enriched_listings = []
      for source, listing in combined_results:
        image_url = None
        match = image_url_pattern.search(listing)
        if match:
          image_url = match.group(0)
        listing_data = {
          "source": source,
          "zip_code": zip_code,
          "keyword": dynamic_params.get("refined_query", search_description),
          "listing_text": listing,
          "image_url": image_url,
        }
        text_emb = get_text_embeddings([listing])
        listing_data["text_embedding"] = text_emb.tolist()[0]
        if image_url:
          img_emb = get_image_embedding(image_url)
          listing_data["image_embedding"] = img_emb.tolist() if img_emb is not None else None
        else:
          listing_data["image_embedding"] = None
        enriched_listings.append(listing_data)
        store_listing_metadata(listing_data)

      st.session_state["listings"] = enriched_listings
      if enriched_listings: # Build index only if there are listings
        text_embeddings = np.array([np.array(d["text_embedding"], dtype="float32") for d in enriched_listings])
        text_index = build_faiss_index(text_embeddings)
        st.session_state["text_faiss_index"] = text_index
      else:
        st.session_state["text_faiss_index"] = None # Handle case with no listings

      num_columns = 3
      cols = st.columns(num_columns)
      for idx, listing in enumerate(enriched_listings):
        with cols[idx % num_columns]:
          render_listing_card(listing)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_search())

# --- Similarity Search Section ---
def run_similarity_search():
  query_text = st.text_input("Enter text to search for similar listings within this session:")
  if query_text and "text_faiss_index" in st.session_state and st.session_state["text_faiss_index"] is not None: # Check if index exists and is not None
    async def do_similarity():
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(do_similarity())
  elif query_text: # if there is a query but no index
    st.warning("No listings have been found in the current session to perform similarity search. Please initiate a search first.")


run_similarity_search()
