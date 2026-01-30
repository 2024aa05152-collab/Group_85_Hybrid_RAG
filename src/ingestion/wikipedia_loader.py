import requests
import json
import os
from typing import List, Dict, Any

class WikipediaLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MyHybridRAGProject/1.0'})

    def fetch_content(self, url: str) -> Dict[str, Any]:
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:

                title = url.split('/')[-1].replace('_', ' ')
                return {
                    "url": url,
                    "title": title,
                    "html_content": response.text,
                    "success": True
                }
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return {"success": False}