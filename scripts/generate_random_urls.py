import wikipediaapi
import random
import json
import time
import requests

# Setup Wikipedia API with a descriptive User Agent
wiki = wikipediaapi.Wikipedia(
    user_agent="MyHybridRAGProject/1.0",
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def get_random_urls(count=300, min_words=301, max_words=500):
    random_urls = [] 
    seen_titles = set()
    retry_count = 0
    max_retries = 2000  
    
    print(f"Starting to sample {count} random Wikipedia URLs...")
    
    while len(random_urls) < count and retry_count < max_retries:
        try:
            headers = {
                'User-Agent': 'MyHybridRAGProject/1.0 (https://en.wikipedia.org/wiki/Wikipedia:User-Agent_policy)'
            }
            response = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                retry_count += 1
                time.sleep(0.5)
                continue
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                retry_count += 1
                time.sleep(0.5)
                continue
            
            title = data.get('title')
            url = data.get('content_urls', {}).get('desktop', {}).get('page')
            
            if not title or not url:
                retry_count += 1
                continue
            
            if title not in seen_titles:
                page = wiki.page(title)
                
                if page.exists() and "may refer to" not in page.summary:
                    word_count = len(page.text.split())
                    
                    if min_words <= word_count <= max_words:
                        random_urls.append(url)  
                        seen_titles.add(title)
                        print(f"[{len(random_urls)}/{count}] Added: {url} ({word_count} words)")
                
                retry_count = 0 
            else:
                retry_count += 1
                
        except Exception as e:
            retry_count += 1
            time.sleep(0.5)

    return random_urls

# Execute and Save
if __name__ == "__main__":
    results = get_random_urls(300, 301, 500)
    
    with open('data/random_urls.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSuccess! {len(results)} random URLs saved to random_urls.json")