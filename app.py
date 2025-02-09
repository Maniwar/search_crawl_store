import streamlit as st
from playwright.async_api import async_playwright
import asyncio

async def run_playwright():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("http://playwright.dev")
        title = await page.title()
        st.write(f"Page title: {title}") # Display the title in Streamlit
        await browser.close()

st.write("Starting the test…")

asyncio.run(run_playwright())
