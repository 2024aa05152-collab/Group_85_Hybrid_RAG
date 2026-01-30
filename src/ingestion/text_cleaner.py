import re
from bs4 import BeautifulSoup

class TextCleaner:
    def clean_html(self, html_content: str) -> str:
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove UI elements and references
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'table', 'sup', 'span']):
            element.decompose()

        # Get main content text
        text = soup.get_text(separator=' ')
        
        # Clean Wikipedia specific artifacts
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Cut off the References/External Links sections
        for marker in ["References", "Bibliography", "External links"]:
            if marker in text:
                text = text.split(marker)[0]
                
        return text

    def validate_text_length(self, text: str, min_words: int = 200) -> bool:
        return len(text.split()) >= min_words