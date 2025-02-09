import os
import subprocess
import streamlit as st
import asyncio
import json
import random
from typing import List
from pydantic import BaseModel

# ----------------------------------------------------------------
# Installation of Playwright Browsers & Dependencies
# ----------------------------------------------------------------
os.system('playwright install')
os.system('playwright install-deps')

 

# ----------------------------------------------------------------
# SECRETS & CONFIGURATION
# ----------------------------------------------------------------
# Your .streamlit/secrets.toml should contain:
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

# ----------------------------------------------------------------
# Supabase Client Initialization
# ----------------------------------------------------------------
from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------------------------------------------
# Crawl4AI Imports and Configuration
# ----------------------------------------------------------------
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# ----------------------------------------------------------------
# OpenAI Client Initialization for Agent (New Interface)
# ----------------------------------------------------------------
# Using the new interface per https://github.com/openai/openai-python.
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
try:
    _ = client.models.list()
    st.write("OpenAI client initialized successfully.")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")

# ----------------------------------------------------------------
# Data Model for Product Data
# ----------------------------------------------------------------
class Product(BaseModel):
    source: str       # Which site the product came from
    url: str          # URL of the product page
    title: str        # Product title
    description: str  # Product description
    price: str        # Price as a string (e.g. "$299")
    image_url: str    # URL to the product image

# ----------------------------------------------------------------
# Extraction Schema for CSS-based Parsing
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# Global Dictionary of Available Sites
# ----------------------------------------------------------------
AVAILABLE_SITES = {
    "EcommerceSite": "https://example-ecommerce.com/search?q={query}",
    "Facebook Marketplace": "https://www.facebook.com/marketplace/search/?query={query}",
    "eBay": "https://www.ebay.com/sch/i.html?_nkw={query}"
}

# ----------------------------------------------------------------
# LLM-based Dynamic Site Selection
# ----------------------------------------------------------------
def choose_sites(query: str) -> List[str]:
    """
    Use the OpenAI client to choose which sites to search based on the query.
    Available sites are: EcommerceSite, Facebook Marketplace, eBay.
    Returns a list of site names.
    """
    prompt = f"""Given the product query: "{query}", decide which of the following sites to search:
EcommerceSite, Facebook Marketplace, eBay.
Return a comma-separated list of site names (exactly as given)."""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that selects e-commerce sites to search."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )
        result_text = completion.choices[0].message["content"].strip()
        # Expect a comma-separated list, e.g., "EcommerceSite, eBay"
        sites = [site.strip() for site in result_text.split(",") if site.strip() in AVAILABLE_SITES]
        if not sites:
            sites = list(AVAILABLE_SITES.keys())  # Fallback to all if LLM doesn't pick any valid site.
        return sites
    except Exception as e:
        st.error(f"Error selecting sites: {e}")
        return list(AVAILABLE_SITES.keys())

# ----------------------------------------------------------------
# Dynamic Target URL Generation
# ----------------------------------------------------------------
def get_target_urls(query: str, sites: List[str]) -> dict:
    """
    Generate search URLs for the selected target sites.
    Dynamically adjust the URLs based on the query.
    If "used" is in the query and the site is Facebook Marketplace or eBay, append "used" to the query.
    """
    query_encoded = query.replace(" ", "+")
    urls = {}
    for site in sites:
        base_url = AVAILABLE_SITES[site]
        if site in ["Facebook Marketplace", "eBay"] and "used" in query.lower():
            final_query = f"{query_encoded}+used"
        else:
            final_query = query_encoded
        urls[site] = base_url.format(query=final_query)
    return urls

# ----------------------------------------------------------------
# Asynchronous Scraping Functions Using CSS Extraction
# ----------------------------------------------------------------
async def scrape_site_css(site: str, url: str) -> List[Product]:
    """
    Scrapes the given URL using Crawl4AI with a CSS-based extraction strategy.
    Returns a list of Product objects.
    """
    # Use default Chromium by not passing an unsupported parameter.
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

async def scrape_all_sites(query: str, sites: List[str]) -> List[Product]:
    """
    Concurrently scrape each selected target site for the query.
    Returns a combined list of Product objects.
    """
    target_urls = get_target_urls(query, sites)
    tasks = [scrape_site_css(site, url) for site, url in target_urls.items()]
    results = await asyncio.gather(*tasks)
    all_products = [prod for sublist in results for prod in sublist]
    return all_products

# ----------------------------------------------------------------
# Supabase Dynamic Schema Functions
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# Agentic Process: LLM Answer Based on Supabase Data
# ----------------------------------------------------------------
def agent_answer(query: str) -> str:
    """
    Retrieves all stored products, builds a context prompt, and uses the OpenAI client
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

# ----------------------------------------------------------------
# Streamlit Chat Interface
# ----------------------------------------------------------------
st.title("Intelligent Shopping Chat with Agentic LLM & Supabase")
st.write("Ask for a product (e.g., 'used iPhone') and the app will scrape multiple sites, store results in Supabase, and then let the agent decide which products are best based on the stored data.")

# Use LLM to choose which sites to search.
user_query = st.text_input("Enter your product query for scraping (e.g., 'used iPhone'):")

if st.button("Choose Sites & Search"):
    if user_query:
        chosen_sites = choose_sites(user_query)
        st.write(f"LLM selected the following sites: {', '.join(chosen_sites)}")
        st.session_state.conversation = st.session_state.get("conversation", [])
        st.session_state.conversation.append({"sender": "user", "message": f"Scrape: {user_query}"})
        with st.spinner("Scraping relevant sites..."):
            all_products = asyncio.run(scrape_all_sites(user_query, chosen_sites))
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
    1. Enter a product query (e.g., "used iPhone", "budget laptop") and click "Choose Sites & Search".
    2. The LLM will select which sites to search, then the app will scrape those sites, extract product details, and store them in Supabase using a dynamic schema.
    
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
