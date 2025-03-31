import xml.etree.ElementTree as ET

try:
    tree = ET.parse("new_sitemap.xml")
    root = tree.getroot()
    for url in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
        print(url.text)
    print("Sitemap parsed successfully!")
except FileNotFoundError:
    print("Error: new_sitemap.xml not found.")
    exit(1)
except Exception as e:
    print(f"Error parsing sitemap: {e}")
    exit(1)