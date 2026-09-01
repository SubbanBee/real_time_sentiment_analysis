from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}


def scrape_google_news(topic: str, max_items: int = 15):
    """
    Fetch publicly accessible Google News RSS title and snippet text.
    No login, authentication bypass, CAPTCHA bypass, or private content scraping.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Please enter a topic or keyword.")

    url = (
        "https://news.google.com/rss/search?q="
        f"{quote_plus(topic.strip())}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            "Could not fetch public news content. Check your internet connection "
            "or try another topic."
        ) from error

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")
    records = []

    for item in items[:max_items]:
        title_tag = item.find("title")
        source_tag = item.find("source")
        description_tag = item.find("description")
        link_tag = item.find("link")

        title = title_tag.get_text(strip=True) if title_tag else ""
        source = source_tag.get_text(strip=True) if source_tag else "Google News"
        description = (
            description_tag.get_text(" ", strip=True)
            if description_tag else ""
        )
        link = link_tag.get_text(strip=True) if link_tag else ""

        text = f"{title}. {description}".strip()

        if len(text) > 10:
            records.append({
                "source": source,
                "text": text,
                "url": link
            })

    if not records:
        raise RuntimeError(
            "No publicly accessible news text was found. Try a different topic."
        )

    return records
