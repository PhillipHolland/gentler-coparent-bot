import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time

# Load sitemap
try:
    tree = ET.parse("new_sitemap.xml")
    root = tree.getroot()
except FileNotFoundError:
    print("Error: new_sitemap.xml not found.")
    exit(1)
except Exception as e:
    print(f"Error parsing sitemap: {e}")
    exit(1)

# Start with empty text
web_text = ""

# Process each URL
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
    page_url = url.text
    try:
        response = requests.get(page_url, timeout=10, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.text, "html.parser")
        web_text += soup.get_text(separator=" ") + "\n"
        print(f"Scraped: {page_url}")
        time.sleep(1)  # 1-second delay between requests
    except Exception as e:
        print(f"Error with {page_url}: {e}")

# Save web content
try:
    with open("web_training_data.txt", "w", encoding='utf-8') as file:
        file.write(web_text)
except Exception as e:
    print(f"Error writing to web_training_data.txt: {e}")
    exit(1)

print("Website content extracted and saved!")