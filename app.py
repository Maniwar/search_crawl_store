import streamlit as st
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from supabase.client import create_client
from openai import OpenAI
import json
from bs4 import BeautifulSoup

# Initialize clients using secrets
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["key"]
openai_api_key = st.secrets["openai_api"]["key"]

def scrape_website(url):
    try:
        browser_config = BrowserConfig(
            headless=True,
            verbose=True
        )
        
        run_config = CrawlerRunConfig(cache_mode="BYPASS")
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url, config=run_config)
            return result.cleaned_html
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
        return None

def save_to_supabase(products):
    try:
        supabase = create_client(supabase_url, supabase_key)
        response = supabase.table('products').insert(products)
        st.success(f"Successfully saved {len(products)} products to database")
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

def search_products(query):
    try:
        supabase = create_client(supabase_url, supabase_key)
        products = supabase.table('products').select(
            "*", 
            count="exact",
            order_by=("name", {"ascending": True}),
            limit=10,
            like=("name", f"%{query}%")
        ).execute()
        
        return products.get("data", [])
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def get_recommendations(query):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": f"Recommend products related to: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return ""

def extract_product_info(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        products = []
        
        # Assuming products are in a list with class 'product-list'
        product_list = soup.find('ul', class_='product-list')
        if not product_list:
            return products
        
        for product_item in product_list.find_all('li'):
            product = {
                'name': product_item.find('h3').text,
                'price': product_item.find('p', class_='price').text,
                'description': product_item.find('p', class_='desc').text,
                'image_url': product_item.find('img')['src']
            }
            products.append(product)
        
        return products
    except Exception as e:
        st.error(f"Data extraction failed: {str(e)}")
        return []

def main():
    st.title("AI Shopping Assistant")
    st.write("A powerful tool to help you find and compare products")

    option = st.selectbox('What do you want to do?', ['Scrape Website', 'Search Products'])

    if option == 'Scrape Website':
        url = st.text_input('Enter website URL')
        if st.button('Scrape'):
            html_content = scrape_website(url)
            if html_content:
                products = extract_product_info(html_content)
                if save_to_supabase(products):
                    st.success("Data saved to database!")
    elif option == 'Search Products':
        query = st.text_input('What are you looking for?')
        if st.button('Search'):
            results = search_products(query)
            recommendations = get_recommendations(query)

            # Display search results
            st.subheader("Search Results")
            if len(results) > 0:
                cols = st.columns(len(results))
                for i, product in enumerate(results):
                    with cols[i]:
                        st.image(product['image_url'], use_column_width=True)
                        st.write(f"**{product['name']}**")
                        st.write(f"Price: {product.get('price', 'N/A')}")
                        st.write(product['description'])
            else:
                st.info("No products found matching your search")

            # Display recommendations
            st.subheader("Recommended Items")
            if len(recommendations) > 0:
                cols = st.columns(len(recommendations))
                for i, recommendation in enumerate(recommendations):
                    with cols[i]:
                        st.image(recommendation['image_url'], use_column_width=True)
                        st.write(f"**{recommendation['name']}**")
                        st.write(f"Price: {recommendation.get('price', 'N/A')}")
                        st.write(recommendation['description'])
            else:
                st.info("No recommendations available")

if __name__ == "__main__":
    main()
