import re
from bs4 import BeautifulSoup
import html2text
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean HTML content from Wikipedia pages"""

    def __init__(self):
        self.html2text_converter = html2text.HTML2Text()
        self.html2text_converter.ignore_links = False
        self.html2text_converter.ignore_images = True
        self.html2text_converter.body_width = 0  # No wrapping

    def clean_html(self, html_content: str) -> str:
        """Convert HTML to clean text"""
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer',
                                          'table', 'sup', 'span']):
                element.decompose()

            # Get main content
            main_content = soup.find('div', {'id': 'mw-content-text'})
            if main_content:
                html_content = str(main_content)

            # Convert to markdown
            text = self.html2text_converter.handle(html_content)

            # Clean up
            text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
            text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spaces
            text = re.sub(r'\[edit\]', '', text)  # Remove edit links
            text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers

            # Remove references section
            lines = text.split('\n')
            clean_lines = []
            in_references = False

            for line in lines:
                if 'references' in line.lower() or 'bibliography' in line.lower():
                    in_references = True
                elif in_references and line.strip() == '':
                    in_references = False

                if not in_references:
                    clean_lines.append(line)

            text = '\n'.join(clean_lines).strip()

            return text

        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return ""

    def validate_text_length(self, text: str, min_words: int = 200) -> bool:
        """Check if text meets minimum word count"""
        words = text.split()
        return len(words) >= min_words

    def process_page(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single page: clean and validate"""
        cleaned_text = self.clean_html(page_data["html_content"])

        result = {
            "url": page_data["url"],
            "title": page_data["title"],
            "original_html": page_data["html_content"],
            "cleaned_text": cleaned_text,
            "word_count": len(cleaned_text.split()),
            "is_valid": self.validate_text_length(cleaned_text, min_words=200)
        }

        return result

    def process_all_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all pages"""
        cleaned_pages = []

        for page in pages:
            cleaned_page = self.process_page(page)
            if cleaned_page["is_valid"]:
                cleaned_pages.append(cleaned_page)

        logger.info(f"Cleaned {len(cleaned_pages)}/{len(pages)} valid pages")
        return cleaned_pages