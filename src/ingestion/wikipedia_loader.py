import json
import random
import requests
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm
import logging
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaLoader:
    """Load and manage Wikipedia URLs with robust error handling"""

    def __init__(self, config):
        self.config = config
        self.fixed_urls_path = config["fixed_urls"]
        self.random_urls_path = config["random_urls"]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def load_fixed_urls(self) -> List[str]:
        """Load fixed URLs from JSON file"""
        try:
            with open(self.fixed_urls_path, 'r') as f:
                urls = json.load(f)
            logger.info(f"Loaded {len(urls)} fixed URLs")
            return urls
        except FileNotFoundError:
            logger.warning(f"Fixed URLs file not found: {self.fixed_urls_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing fixed URLs file: {e}")
            return []

    def generate_random_urls(self, n: int = 300) -> List[str]:
        """Generate random Wikipedia URLs with fallback to predefined list"""

        # Comprehensive list of Wikipedia pages as fallback
        fallback_pages = [
            "Artificial_intelligence", "Machine_learning", "Deep_learning",
            "Python_(programming_language)", "JavaScript", "Java_(programming_language)",
            "United_States", "India", "China", "Japan", "Germany", "United_Kingdom",
            "World_War_II", "World_War_I", "Cold_War", "French_Revolution",
            "Climate_change", "Global_warming", "Renewable_energy", "Solar_power",
            "Quantum_mechanics", "Relativity", "Big_Bang", "Black_hole",
            "Biology", "Chemistry", "Physics", "Mathematics", "Computer_science",
            "Psychology", "Philosophy", "Economics", "Political_science",
            "Literature", "Shakespeare", "Poetry", "Novel",
            "Music", "Classical_music", "Jazz", "Rock_music",
            "Art", "Painting", "Sculpture", "Architecture",
            "Sports", "Football", "Basketball", "Cricket", "Tennis",
            "Technology", "Internet", "Smartphone", "Electric_car",
            "Health", "Medicine", "Nutrition", "Exercise", "Yoga",
            "Environment", "Ecosystem", "Biodiversity", "Conservation",
            "Space_exploration", "NASA", "International_Space_Station", "Moon",
            "History", "Ancient_history", "Middle_Ages", "Modern_history",
            "Geography", "Earth", "Ocean", "Mountain", "River",
            "Culture", "Language", "Religion", "Tradition",
            "Business", "Marketing", "Finance", "Entrepreneurship",
            "Education", "University", "School", "Learning"
        ]

        random_urls = []
        base_url = "https://en.wikipedia.org/wiki/"
        api_url = "https://en.wikipedia.org/w/api.php"

        # Shuffle fallback pages for variety
        random.shuffle(fallback_pages)

        logger.info(f"Generating {n} random Wikipedia URLs...")

        for i in tqdm(range(n), desc="Generating random URLs"):
            try:
                # Try to get random page from Wikipedia API
                params = {
                    "action": "query",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": 1,
                    "format": "json",
                    "utf8": 1
                }

                response = self.session.get(api_url, params=params, timeout=10)

                # Check if response is valid JSON
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "query" in data and "random" in data["query"]:
                            page_title = data["query"]["random"][0]["title"]
                            # Clean the title for URL
                            page_title = urllib.parse.quote(page_title.replace(" ", "_"))
                            url = f"{base_url}{page_title}"
                            random_urls.append(url)
                        else:
                            raise ValueError("Invalid API response structure")
                    except json.JSONDecodeError:
                        # Fallback to predefined page
                        fallback_title = fallback_pages[i % len(fallback_pages)]
                        url = f"{base_url}{fallback_title}"
                        random_urls.append(url)
                else:
                    # Fallback to predefined page
                    fallback_title = fallback_pages[i % len(fallback_pages)]
                    url = f"{base_url}{fallback_title}"
                    random_urls.append(url)

                # Be polite - small delay between requests
                time.sleep(0.1)

            except Exception as e:
                logger.debug(f"Failed to get random URL (attempt {i + 1}): {e}")
                # Fallback to predefined page
                fallback_title = fallback_pages[i % len(fallback_pages)]
                url = f"{base_url}{fallback_title}"
                random_urls.append(url)

        # Remove duplicates while preserving order
        random_urls = list(dict.fromkeys(random_urls))

        # If we still don't have enough unique URLs, add more from fallback
        while len(random_urls) < n:
            extra_pages = [p for p in fallback_pages if p not in random_urls]
            if extra_pages:
                for page in extra_pages[:n - len(random_urls)]:
                    random_urls.append(f"{base_url}{page}")
            else:
                # Add numbered pages if we run out
                for j in range(n - len(random_urls)):
                    random_urls.append(f"{base_url}Template:{j}")

        # Save random URLs
        with open(self.random_urls_path, 'w') as f:
            json.dump(random_urls[:n], f, indent=2)

        logger.info(f"Generated {len(random_urls)} random URLs")
        return random_urls[:n]

    def fetch_wikipedia_content(self, url: str) -> Dict[str, Any]:
        """Fetch and parse Wikipedia page content with multiple fallbacks"""
        try:
            # Extract page title from URL
            if '/wiki/' in url:
                page_title = url.split('/wiki/')[-1]
            else:
                page_title = url.split('/')[-1]

            # Decode URL encoding
            page_title = urllib.parse.unquote(page_title)

            logger.debug(f"Fetching: {page_title}")

            # Try different methods to get content

            # Method 1: Wikipedia API with parse action
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "parse",
                "page": page_title.replace('_', ' '),
                "format": "json",
                "prop": "text|displaytitle",
                "section": 0,
                "disabletoc": 1,
                "disableeditsection": 1,
                "disablelimitreport": 1,
                "utf8": 1
            }

            response = self.session.get(api_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "parse" in data:
                    html_content = data["parse"]["text"]["*"]
                    title = data["parse"]["displaytitle"]

                    return {
                        "url": url,
                        "title": title,
                        "html_content": html_content,
                        "success": True,
                        "method": "api_parse"
                    }

            # Method 2: Direct page fetch with simpler extraction
            logger.debug(f"API parse failed for {page_title}, trying direct fetch...")

            direct_response = self.session.get(url, timeout=30)
            if direct_response.status_code == 200:
                # Extract main content using simple text extraction
                content = direct_response.text

                # Simple extraction of main content (between <div id="mw-content-text"> tags)
                if '<div id="mw-content-text"' in content:
                    start_idx = content.find('<div id="mw-content-text"')
                    end_idx = content.find('</div>', start_idx)
                    if end_idx > start_idx:
                        html_content = content[start_idx:end_idx]

                        # Extract title
                        if '<h1 id="firstHeading"' in content:
                            title_start = content.find('<h1 id="firstHeading"')
                            title_end = content.find('</h1>', title_start)
                            if title_end > title_start:
                                title_html = content[title_start:title_end]
                                # Extract text from title
                                if '>' in title_html:
                                    title = title_html.split('>', 1)[1].split('<')[0]
                                else:
                                    title = page_title.replace('_', ' ')
                        else:
                            title = page_title.replace('_', ' ')

                        return {
                            "url": url,
                            "title": title,
                            "html_content": html_content,
                            "success": True,
                            "method": "direct_fetch"
                        }

            # Method 3: Return minimal content
            logger.debug(f"Both methods failed for {page_title}, returning minimal content...")
            return {
                "url": url,
                "title": page_title.replace('_', ' '),
                "html_content": f"<div><h1>{page_title.replace('_', ' ')}</h1><p>Content could not be fetched.</p></div>",
                "success": False,
                "error": "Failed to fetch content",
                "method": "fallback"
            }

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return {
                "url": url,
                "title": url.split('/wiki/')[-1].replace('_', ' ') if '/wiki/' in url else "Unknown",
                "html_content": "",
                "success": False,
                "error": str(e),
                "method": "error"
            }

    def fetch_all_contents(self, urls: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Fetch content for all URLs with concurrency control"""
        contents = []

        for i, url in enumerate(tqdm(urls, desc="Fetching Wikipedia pages")):
            content = self.fetch_wikipedia_content(url)
            contents.append(content)

            # Rate limiting - be polite to Wikipedia servers
            time.sleep(0.2)  # 5 requests per second max

            # Log progress every 10 pages
            if (i + 1) % 10 == 0:
                successful = sum(1 for c in contents if c["success"])
                logger.info(f"Fetched {i + 1}/{len(urls)} pages, {successful} successful")

        # Filter out failed fetches
        successful_contents = [c for c in contents if c["success"]]
        logger.info(f"Successfully fetched {len(successful_contents)}/{len(urls)} pages")

        # Log methods used
        methods = {}
        for c in contents:
            method = c.get("method", "unknown")
            methods[method] = methods.get(method, 0) + 1

        for method, count in methods.items():
            logger.info(f"  - {method}: {count} pages")

        return successful_contents

    def get_all_urls(self, regenerate_random: bool = False) -> List[str]:
        """Get all 500 URLs (200 fixed + 300 random)"""
        fixed_urls = self.load_fixed_urls()

        # If we don't have enough fixed URLs, generate some
        if len(fixed_urls) < 200:
            logger.warning(f"Only {len(fixed_urls)} fixed URLs found, need 200")
            # Add some default URLs
            default_urls = [
                "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "https://en.wikipedia.org/wiki/United_States",
                "https://en.wikipedia.org/wiki/World_War_II",
                "https://en.wikipedia.org/wiki/Climate_change",
                "https://en.wikipedia.org/wiki/Quantum_mechanics",
                "https://en.wikipedia.org/wiki/Renewable_energy",
                "https://en.wikipedia.org/wiki/History_of_computing",
                "https://en.wikipedia.org/wiki/Biology"
            ]
            fixed_urls = default_urls + fixed_urls
            fixed_urls = fixed_urls[:200]  # Limit to 200

        if regenerate_random or not self.random_urls_path.exists():
            random_urls = self.generate_random_urls(300)
        else:
            try:
                with open(self.random_urls_path, 'r') as f:
                    random_urls = json.load(f)
            except:
                random_urls = self.generate_random_urls(300)

        all_urls = fixed_urls + random_urls
        logger.info(f"Total URLs: {len(all_urls)} (Fixed: {len(fixed_urls)}, Random: {len(random_urls)})")
        return all_urls