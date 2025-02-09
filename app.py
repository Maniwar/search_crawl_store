import os

# Install Playwright browsers and dependencies at runtime.
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
# Ensure your .streamlit/secrets.toml contains:
#
# [supabase]
# url = "https://your-supabase-url.supabase.co"
# key = "your_supabase_key_here"
#
# [crawl4ai]
# openai_api_key = "your_openai_api_key_here"
#
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
OPENAI_API_KEY = st.secrets["crawl4ai"].get("openai_api_key")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not defined in your secrets! Please add it under [crawl4ai].")

# --------------------------------------------------
# Supabase Client Initialization
# --------------------------------------------------
from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------
# Crawl4AI Imports and Configuration
# --------------------------------------------------
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# --------------------------------------------------
# OpenAI Client Initialization for Agent (New Interface)
# --------------------------------------------------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
# Optionally, test the client:
try:
    _ = client.models.list()
    st.write("OpenAI client initialized successfully.")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")

# --------------------------------------------------
# Data Model for Product Data
# --------------------------------------------------
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
    Generate search URLs for different target sites.
    Dynamically adjust the URLs based on the query. For example, if "used" is in the query,
    append "used" to the search parameters for Facebook Marketplace and eBay.
    Replace these example URLs with your actual target URLs.
    """
    query_encoded = query.replace(" ", "+")
    urls = {}
    urls["EcommerceSite"] = f"https://example-ecommerce.com/search?q={query_encoded}"
    if "used" in query.lower():
        urls["Facebook Marketplace"] = f"https://www.facebook.com/marketplace/search/?query={query_encoded}+used"
        urls["eBay"] = f"https://www.ebay.com/sch/i.html?_nkw={query_encoded}+used"
    else:
        urls["Facebook Marketplace"] = f"https://www.facebook.com/marketplace/search/?query={query_encoded}"
        urls["eBay"] = f"https://www.ebay.com/sch/i.html?_nkw={query_encoded}"
    return urls

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
            try:
                extracted = json.loads(result.extracted_content)
            except Exception as e:
                st.error(f"Error parsing JSON from {site}: {e}")
                extracted = []
            if not isinstance(extracted, list) or not extracted:
                st.warning(f"No product data extracted from {site}.")
                return []
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
    Concurrently scrape each target site for the query.
    Returns a combined list of Product objects.
    """
    target_urls = get_target_urls(query)
    tasks = [scrape_site_css(site, url) for site, url in target_urls.items()]
    results = await asyncio.gather(*tasks)
    all_products = [prod for sublist in results for prod in sublist]
    return all_products

# --------------------------------------------------
# Supabase Dynamic Schema Functions
# --------------------------------------------------
def insert_product_to_supabase_dynamic(product: Product):
    """
    Inserts a product record into Supabase using a dynamic JSONB schema.
    """
    data = {
        "source": product.source,
        "data": product.model_dump()  # Using model_dump() per Pydantic V2
    }
    response = supabase.table("products").insert(data).execute()
    try:
        if response.error:
            st.error(f"Error inserting product: {response.error.message}")
        else:
            st.success("Product inserted successfully!")
    except AttributeError:
        st.success("Product inserted (no error attribute available).")

def get_all_products_dynamic() -> List[Product]:
    """
    Retrieves all product records from Supabase.
    Converts the JSONB 'data' column back into Product objects.
    """
    response = supabase.table("products").select("*").execute()
    products = []
    try:
        if response.error:
            st.error(f"Error retrieving products: {response.error.message}")
            return []
    except AttributeError:
        pass
    for record in response.data:
        try:
            prod = Product(**record["data"])
            products.append(prod)
        except Exception as e:
            st.error(f"Error parsing product record: {e}")
    return products

# --------------------------------------------------
# Agentic Process: LLM Answer Based on Supabase Data
# --------------------------------------------------
def agent_answer(query: str) -> str:
    """
    Retrieves all stored products, builds a context prompt, and uses the new OpenAI client
    to answer the user's query about which products are good.
    """
    products = get_all_products_dynamic()
    if not products:
        return "No product data available to answer your query."
    
    # Build context from product records.
    context_lines = []
    for prod in products:
        line = f"{prod.source}: {prod.title} at {prod.price}. {prod.description}"
        context_lines.append(line)
    context_text = "\n".join(context_lines)
    
    prompt = f"""Based on the following product information:
{context_text}

Answer the following query: {query}

Which products are the best and why?"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intelligent shopping assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        answer = completion.choices[0].message
        return answer
    except Exception as e:
        return f"Error generating agent answer: {e}"

# --------------------------------------------------
# Streamlit Chat Interface
# --------------------------------------------------
st.title("Intelligent Shopping Chat with Agentic LLM & Supabase")
st.write("Ask for a product (e.g., 'used iPhone') and the app will scrape multiple sites and store results in Supabase. Then, ask the agent which products are best based on the stored data.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display conversation with HTML formatting.
for msg in st.session_state.conversation:
    if msg["sender"] == "user":
        st.markdown(f"**You:** {msg['message']}", unsafe_allow_html=True)
    else:
        st.markdown(f"**Assistant:** {msg['message']}", unsafe_allow_html=True)

# Section for scraping & storing products.
user_query = st.text_input("Enter your product query for scraping (e.g., 'used iPhone'):")

if st.button("Search & Store Products"):
    if user_query:
        st.session_state.conversation.append({"sender": "user", "message": f"Scrape: {user_query}"})
        with st.spinner("Scraping relevant sites..."):
            all_products = asyncio.run(scrape_all_sites(user_query))
        for prod in all_products:
            insert_product_to_supabase_dynamic(prod)
        if all_products:
            response_message = "I found and stored the following products:<br><br>"
            for prod in all_products:
                response_message += (
                    f"**{prod.source}:** <a href='{prod.url}' target='_blank'>{prod.title}</a> - {prod.price}<br>"
                    f"{prod.description}<br><br>"
                )
        else:
            response_message = "Sorry, I couldn't find any products matching your query."
        st.session_state.conversation.append({"sender": "assistant", "message": response_message})
        st.rerun()
    else:
        st.warning("Please enter a product query.")

# Section for agent queries.
agent_query = st.text_input("Enter your agent query (e.g., 'What is the best product and why?'):")

if st.button("Ask Agent"):
    if agent_query:
        st.session_state.conversation.append({"sender": "user", "message": f"Agent Query: {agent_query}"})
        with st.spinner("Letting the agent analyze stored products..."):
            answer = agent_answer(agent_query)
        st.session_state.conversation.append({"sender": "assistant", "message": answer})
        st.rerun()
    else:
        st.warning("Please enter an agent query.")

# Option to view all stored products.
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

st.sidebar.markdown("### Instructions")
st.sidebar.markdown(
    """
    **Scraping & Storing Products:**
    1. Enter a product query (e.g., "used iPhone", "budget laptop") and click "Search & Store Products".
    2. The app scrapes multiple sites (using dynamic URLs based on your query), extracts product details, and stores them in Supabase using a dynamic schema.
    
    **Agent Query:**
    1. After products are stored, enter an agent query (e.g., "What is the best product and why?").
    2. The agent retrieves stored products, builds context from their details, and uses OpenAI to answer your query.
    
    **Viewing Products:**
    1. Click "View Stored Products" to see all products currently stored in Supabase.
    
    **Note:**
    - Adjust the extraction schema and target URLs for your actual use case.
    - Ensure your Supabase table is set up with a JSONB column (named "data") to support the dynamic schema.
    """
)
