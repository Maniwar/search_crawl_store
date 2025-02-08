import os
import subprocess

# Optionally disable Streamlit's file watcher to avoid certain Torch errors
os.environ["STREAMLIT_WATCHER_DISABLED"] = "true"

import streamlit as st
st.set_page_config(page_title="Local Listings Shopping Session", layout="wide")

# --- Attempt to install Playwright browsers from the shell script ---
if os.path.exists("install_browsers.sh"):
    try:
        subprocess.run(["bash", "install_browsers.sh"], check=True)
        st.info("Playwright browsers installed successfully.")
    except subprocess.CalledProcessError as e:
        st.warning(f"Playwright installation command failed: {e}")
    except Exception as e:
        st.warning(f"Unexpected error during Playwright installation: {e}")
else:
    st.info("install_browsers.sh not found; skipping browser installation.")

# Now import and run your main logic
import re
import asyncio
import json
from io import BytesIO
from urllib.parse import urlencode
import numpy as np
import requests
from PIL import Image
import nest_asyncio
import faiss
import torch
from supabase import create_client, Client
from transformers import CLIPProcessor, CLIPModel
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from openai import OpenAI

# ... The rest of your logic (similar to what you already have) ...
# See your posted code for generating searches, crawling, classification, embeddings, etc.

# Example of final search button, just keep your existing logic:
if st.button("Search Listings"):
    # ...
    pass

# And a similarity search function:
def run_similarity_search():
    # ...
    pass

run_similarity_search()
