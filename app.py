import os

os.system('playwright install')
os.system('playwright install-deps')

import subprocess

try:
    subprocess.run(["python", "-m", "playwright", "install"], check=True)
except Exception as e:
    print(f"Playwright installation failed: {e}")

import streamlit as st
import asyncio
import json
import random
from typing import List
from pydantic import BaseModel

# --------------------------------------------------
# SECRETS & CONFIGURATION
# --------------------------------------------------
# In your Streamlit Cloud secrets file (.streamlit/secrets.toml), add:
#
# [supabase]
# url = "https://your-supabase-url.supabase.co"
# key = "your_supabase_key_here"
#
# [crawl4ai]
# openai_api_key = "your_openai_api_key_here"  # if needed for LLM-based extraction

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

# --------------------------------------------------
# Supabase Client Initialization
# --------------------------------------------------
# Ensure you have installed supabase-py (pip install supabase)
from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# Crawl4AI Imports and Configuration
# --------------------------------------------------
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# --------------------------------------------------
# Data Model for Product Data
# --------------------------------------------------
# This Pydantic model defines the fields we expect to extract.
class Product(BaseModel):
    source: str       # Which site the product came from
    url: str          # URL of the product page
    title: str        # Product title
    description: str  # Product description
    price: str        # Price as a string (e.g. "$299")
    image_url: str    # URL to the product image

# --------------------------------------------------
# Extraction Schema for CSS-based Parsing
# --------------------------------------------------
# For demonstration, we use a generic schema.
# In production, each target site might require its own schema.
product_schema = {
    "name": "Product",
    "baseSelector": "div.product",  # Assumes each product is wrapped in <div class="product">
    "fields": [
        {"name": "title", "selector": "h2", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        {"name": "description", "selector": "p", "type": "text"},
        {"name": "image_url", "selector": "img", "type": "attribute", "attribute": "src"},
        {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"}
    ]
}

# --------------------------------------------------
# Dynamic Target URL Generation
# --------------------------------------------------
def get_target_urls(query: str) -> dict:
    """
    Given a query string, generate search URLs for different target sites.
    In a real application, these URLs would be tailored to each target site.
    """
    return {
        "EcommerceSite": f"https://example-ecommerce.com/search?q={query}",
        "Facebook Marketplace": f"https://www.facebook.com/marketplace/search/?query={query}",
        "eBay": f"https://www.ebay.com/sch/i.html?_nkw={query}"
    }

# --------------------------------------------------
# Asynchronous Scraping Functions Using CSS Extraction
# --------------------------------------------------
async def scrape_site_css(site: str, url: str) -> List[Product]:
    """
    Scrapes the given URL using Crawl4AI with a CSS-based extraction strategy.
    Returns a list of Product objects.
    """
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(product_schema)
    )
    
    products: List[Product] = []
    try:
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=url, config=run_conf)
            # result.extracted_content should be a JSON string
            try:
                extracted = json.loads(result.extracted_content)
            except Exception as e:
                st.error(f"Error parsing JSON from {site}: {e}")
                extracted = []
            if not isinstance(extracted, list) or not extracted:
                # Simulate dummy output for demonstration.
                extracted = [{
                    "title": f"Sample {site} Product",
                    "price": f"${round(random.uniform(50, 500), 2)}",
                    "description": f"A sample product from {site}.",
                    "image_url": "https://via.placeholder.com/150",
                    "url": url + "/product/sample"
                }]
            for item in extracted:
                try:
                    prod = Product(source=site, **item)
                    products.append(prod)
                except Exception as e:
                    st.error(f"Error creating Product from {site}: {e}")
    except Exception as e:
        st.error(f"Error scraping {site}: {e}")
    return products

async def scrape_all_sites(query: str) -> List[Product]:
    """
    Generates target URLs for the query and concurrently scrapes each site.
    Returns a combined list of Product objects.
    """
    target_urls = get_target_urls(query)
    tasks = []
    for site, url in target_urls.items():
        tasks.append(scrape_site_css(site, url))
    results = await asyncio.gather(*tasks)
    all_products = [prod for sublist in results for prod in sublist]
    return all_products

# --------------------------------------------------
# Supabase Dynamic Schema Functions
# --------------------------------------------------
def insert_product_to_supabase_dynamic(product: Product):
    """
    Inserts a product record into the Supabase 'products' table using a dynamic schema.
    The full product details are stored in a JSONB column named "data".
    """
    data = {
        "source": product.source,
        "data": product.dict()  # Entire product as JSON
    }
    response = supabase.table("products").insert(data).execute()
    if response.error:
        st.error(f"Error inserting product: {response.error.message}")
    else:
        st.success("Product inserted successfully!")

def get_all_products_dynamic() -> List[Product]:
    """
    Retrieves all product records from the Supabase 'products' table.
    Converts the JSONB 'data' column back into Product objects.
    """
    response = supabase.table("products").select("*").execute()
    if response.error:
        st.error(f"Error retrieving products: {response.error.message}")
        return []
    products = []
    for record in response.data:
        try:
            # Expect the dynamic product data to be stored in record["data"]
            prod = Product(**record["data"])
            products.append(prod)
        except Exception as e:
            st.error(f"Error parsing product record: {e}")
    return products

# --------------------------------------------------
# Streamlit Chat Interface
# --------------------------------------------------
st.title("Intelligent Shopping Chat with Dynamic Scraping & Supabase")
st.write("Ask for a product (e.g., 'used iPhone') and the app will scrape multiple sites—including used item sources like Facebook Marketplace—and store results in Supabase using a dynamic schema.")

# Initialize conversation history.
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.markdown("### Conversation")
for msg in st.session_state.conversation:
    if msg["sender"] == "user":
        st.markdown(f"**You:** {msg['message']}")
    else:
        st.markdown(f"**Assistant:** {msg['message']}")

# User input for product query.
user_query = st.text_input("Enter your product query (e.g., 'used iPhone', 'budget laptop'):")

if st.button("Search"):
    if user_query:
        st.session_state.conversation.append({"sender": "user", "message": user_query})
        with st.spinner("Scraping relevant sites..."):
            all_products = asyncio.run(scrape_all_sites(user_query))
        # Insert each scraped product into Supabase using dynamic schema.
        for prod in all_products:
            insert_product_to_supabase_dynamic(prod)
        # Build a response message.
        if all_products:
            response_message = "I found the following products:<br><br>"
            for prod in all_products:
                response_message += (
                    f"**{prod.source}:** <a href='{prod.url}' target='_blank'>{prod.title}</a> - {prod.price}<br>"
                    f"{prod.description}<br><br>"
                )
        else:
            response_message = "Sorry, I couldn't find any products matching your query."
        st.session_state.conversation.append({"sender": "assistant", "message": response_message})
        st.experimental_rerun()
    else:
        st.warning("Please enter a product query.")

# Option to view all products stored in Supabase.
if st.button("View Stored Products"):
    products = get_all_products_dynamic()
    if products:
        view_message = "Stored products:<br><br>"
        for prod in products:
            view_message += (
                f"**{prod.source}:** <a href='{prod.url}' target='_blank'>{prod.title}</a> - {prod.price}<br>"
                f"{prod.description}<br><br>"
            )
        st.markdown(view_message, unsafe_allow_html=True)
    else:
        st.info("No products found in the database.")

# --------------------------------------------------
# Sidebar Instructions
# --------------------------------------------------
st.sidebar.markdown("### Instructions")
st.sidebar.markdown(
    """
    1. **Enter a Query:**  
       - Type the product you are looking for (e.g., "used iPhone", "budget laptop").
    
    2. **Dynamic Scraping:**  
       - The app builds search URLs for multiple sites (simulated for EcommerceSite, Facebook Marketplace, and eBay).
       - It uses Crawl4AI with a CSS-based extraction strategy to scrape product details.
    
    3. **Dynamic Schema for Supabase:**  
       - Each scraped product is inserted into Supabase in a dynamic way by storing the full product data as JSON.
       - This allows the schema to adapt if product fields change.
    
    4. **Results Display:**  
       - Products are shown in a chat-like interface with clickable links, prices, and descriptions.
    
    **Note:**  
    - Adjust the extraction schema for each site’s HTML structure.
    - Ensure your Supabase table is set up with a JSONB column (named "data") to support the dynamic schema.
    """
)
