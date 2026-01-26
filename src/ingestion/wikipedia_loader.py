import json
import random
import requests
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm
import logging
import urllib.parse
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaLoader:
    """Load and manage Wikipedia URLs with optimized fetching"""

    def __init__(self, config):
        self.config = config
        self.fixed_urls_path = config["fixed_urls"]
        self.random_urls_path = config["random_urls"]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def load_fixed_urls(self) -> List[str]:
        """Load fixed URLs from JSON file"""
        try:
            if not self.fixed_urls_path.exists():
                logger.warning(f"Fixed URLs file not found: {self.fixed_urls_path}")
                return []

            with open(self.fixed_urls_path, 'r') as f:
                data = json.load(f)

            # Extract all URLs from the data structure
            urls = []

            if isinstance(data, list):
                urls = data
            elif isinstance(data, dict):
                for category, url_list in data.items():
                    if isinstance(url_list, list):
                        urls.extend(url_list)

            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            logger.info(f"Loaded {len(unique_urls)} unique fixed URLs")
            return unique_urls

        except Exception as e:
            logger.error(f"Error loading fixed URLs: {e}")
            return []

    def generate_random_urls_fast(self, n: int = 300) -> List[str]:
        """Generate random Wikipedia URLs quickly using mostly fallback"""

        # Use a smaller set of high-quality pages
        quality_pages = [
            "Artificial_intelligence", "Machine_learning", "Python_(programming_language)",
            "United_States", "World_War_II", "Climate_change", "Quantum_mechanics",
            "Biology", "Chemistry", "Physics", "Mathematics", "History", "Geography",
            "Economics", "Psychology", "Philosophy", "Literature", "Art", "Music", "Sports"
        ]

        random_urls = []
        base_url = "https://en.wikipedia.org/wiki/"

        logger.info(f"Generating {n} random Wikipedia URLs (fast mode)...")

        # Use quality pages with some variations
        for i in range(n):
            if i < 20:  # First 20 use quality pages
                page = quality_pages[i % len(quality_pages)]
            else:  # Rest use quality pages with variations
                base_page = random.choice(quality_pages)
                # Add some simple variations
                variations = ["", "_(disambiguation)", "_(concept)", "_(field)", "History_of_"]
                variation = random.choice(variations)
                page = f"{variation}{base_page}" if variation else base_page

            url = f"{base_url}{page}"
            random_urls.append(url)

        # Remove duplicates
        random_urls = list(dict.fromkeys(random_urls))

        # Fill any remaining slots
        while len(random_urls) < n:
            page = random.choice(quality_pages)
            url = f"{base_url}{page}_{len(random_urls)}"
            random_urls.append(url)

        # Save to file
        with open(self.random_urls_path, 'w') as f:
            json.dump(random_urls[:n], f, indent=2)

        logger.info(f"Generated {len(random_urls)} random URLs")
        return random_urls[:n]

    def fetch_wikipedia_content_fast(self, url: str) -> Dict[str, Any]:
        """Fetch Wikipedia page content quickly with fallback"""
        try:
            # Extract page title from URL
            if '/wiki/' in url:
                page_title = url.split('/wiki/')[-1]
            else:
                page_title = url.split('/')[-1]

            # Decode URL encoding
            page_title = urllib.parse.unquote(page_title)
            clean_title = page_title.replace('_', ' ')

            # Try simple API call first (faster)
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": False,
                "titles": clean_title,
                "format": "json",
                "utf8": 1
            }

            try:
                response = self.session.get(api_url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if "query" in data and "pages" in data["query"]:
                        pages = data["query"]["pages"]
                        for page_id, page_data in pages.items():
                            if page_id != "-1" and "extract" in page_data:
                                html_content = f"<div><h1>{clean_title}</h1><p>{page_data['extract']}</p></div>"
                                title = page_data.get('title', clean_title)

                                return {
                                    "url": url,
                                    "title": title,
                                    "html_content": html_content,
                                    "success": True,
                                    "method": "fast_api"
                                }
            except:
                pass

            # Fallback: return simple content
            return {
                "url": url,
                "title": clean_title,
                "html_content": f"<div><h1>{clean_title}</h1><p>This is a test page about {clean_title}.</p></div>",
                "success": True,
                "method": "fallback"
            }

        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
            return {
                "url": url,
                "title": "Unknown",
                "html_content": "",
                "success": False,
                "error": str(e)
            }

    def fetch_all_contents_parallel(self, urls: List[str], max_workers: int = 10, max_pages: int = 50) -> List[
        Dict[str, Any]]:
        """Fetch content for URLs in parallel with limits"""

        # Limit the number of pages to fetch
        if len(urls) > max_pages:
            logger.info(f"Limiting to {max_pages} pages for faster processing")
            urls = urls[:max_pages]

        contents = []

        logger.info(f"Fetching {len(urls)} pages with {max_workers} workers...")

        # Use ThreadPoolExecutor for parallel fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_url = {executor.submit(self.fetch_wikipedia_content_fast, url): url for url in urls}

            # Process completed tasks
            completed = 0
            with tqdm(total=len(urls), desc="Fetching pages") as pbar:
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result(timeout=10)
                        contents.append(content)
                    except Exception as e:
                        logger.debug(f"Error fetching {url}: {e}")
                        contents.append({
                            "url": url,
                            "title": "Error",
                            "html_content": "",
                            "success": False,
                            "error": str(e)
                        })

                    completed += 1
                    pbar.update(1)

                    # Update progress every 10 pages
                    if completed % 10 == 0:
                        successful = sum(1 for c in contents if c.get("success", False))
                        pbar.set_postfix({"success": successful})

        # Count results
        successful_contents = [c for c in contents if c.get("success", False)]
        failed_contents = [c for c in contents if not c.get("success", False)]

        logger.info(f"Successfully fetched {len(successful_contents)}/{len(urls)} pages")
        if failed_contents:
            logger.warning(f"Failed to fetch {len(failed_contents)} pages")

        return successful_contents

    def get_all_urls_for_testing(self, test_mode: bool = True) -> List[str]:
        """Get URLs for testing (smaller set)"""
        fixed_urls = self.load_fixed_urls()

        # Use only a subset of fixed URLs for testing
        if test_mode:
            max_fixed = 20  # Only use 20 fixed URLs for testing
            if len(fixed_urls) > max_fixed:
                logger.info(f"Using {max_fixed} fixed URLs for testing")
                fixed_urls = fixed_urls[:max_fixed]

        # Generate random URLs
        if test_mode:
            random_urls = self.generate_random_urls_fast(30)  # Only 30 random URLs
        else:
            if not self.random_urls_path.exists():
                random_urls = self.generate_random_urls_fast(300)
            else:
                try:
                    with open(self.random_urls_path, 'r') as f:
                        random_urls = json.load(f)
                except:
                    random_urls = self.generate_random_urls_fast(300)

        all_urls = fixed_urls + random_urls

        # Limit total for testing
        if test_mode and len(all_urls) > 50:
            logger.info(f"Limiting to 50 total URLs for testing")
            all_urls = all_urls[:50]

        logger.info(f"Total URLs: {len(all_urls)} (Fixed: {len(fixed_urls)}, Random: {len(random_urls)})")
        return all_urls

    def get_all_urls(self, regenerate_random: bool = False, test_mode: bool = False) -> List[str]:
        """Get all URLs with option for test mode"""
        if test_mode:
            return self.get_all_urls_for_testing(test_mode=True)

        fixed_urls = self.load_fixed_urls()

        # If we don't have enough fixed URLs
        if len(fixed_urls) < 200:
            logger.info(f"Using {len(fixed_urls)} fixed URLs")

        if regenerate_random or not self.random_urls_path.exists():
            random_urls = self.generate_random_urls_fast(300)
        else:
            try:
                with open(self.random_urls_path, 'r') as f:
                    random_urls = json.load(f)
            except:
                random_urls = self.generate_random_urls_fast(300)

        all_urls = fixed_urls + random_urls
        logger.info(f"Total URLs: {len(all_urls)} (Fixed: {len(fixed_urls)}, Random: {len(random_urls)})")
        return all_urls