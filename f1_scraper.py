import os
import asyncio
import argparse
import logging
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Set, Optional, Any
from rich.console import Console
from rich.progress import Progress
from playwright.async_api import async_playwright, TimeoutError
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import our custom F1AI class
from f1_ai import F1AI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Load environment variables
load_dotenv()

class F1Scraper:
    def __init__(self, max_pages: int = 100, depth: int = 2, f1_ai: Optional[F1AI] = None):
        """
        Initialize the F1 web scraper.
        
        Args:
            max_pages (int): Maximum number of pages to scrape
            depth (int): Maximum depth for crawling
            f1_ai (F1AI): Optional F1AI instance to use for ingestion
        """
        self.max_pages = max_pages
        self.depth = depth
        self.visited_urls: Set[str] = set()
        self.f1_urls: List[str] = []
        self.f1_ai = f1_ai if f1_ai else F1AI(llm_provider="openrouter")
        
        # Define F1-related keywords to identify relevant pages
        self.f1_keywords = [
            "formula 1", "formula one", "f1", "grand prix", "gp", "race", "racing",
            "driver", "team", "championship", "qualifying", "podium", "ferrari",
            "mercedes", "red bull", "mclaren", "williams", "alpine", "aston martin",
            "haas", "alfa romeo", "alphatauri", "fia", "pirelli", "drs", "pit stop",
            "verstappen", "hamilton", "leclerc", "sainz", "norris", "perez",
            "russell", "alonso", "track", "circuit", "lap", "pole position"
        ]
        
        # Core F1 websites to target
        self.f1_core_sites = [
            "formula1.com",
            "autosport.com",
            "motorsport.com",
            "f1i.com",
            "racefans.net",
            "crash.net/f1",
            "espn.com/f1",
            "bbc.com/sport/formula1",
            "skysports.com/f1"
        ]
    
    def is_f1_related(self, url: str, content: Optional[str] = None) -> bool:
        """
        Check if a URL or its content is F1-related.
        
        Args:
            url (str): URL to check
            content (str, optional): Page content to analyze
            
        Returns:
            bool: True if F1-related, False otherwise
        """
        # Check if URL is from a core F1 site
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        for core_site in self.f1_core_sites:
            if core_site in domain:
                return True
        
        # Check URL path for F1 keywords
        url_path = parsed_url.path.lower()
        for keyword in self.f1_keywords:
            if keyword in url_path:
                return True
        
        # If content provided, check for F1 keywords
        if content:
            content_lower = content.lower()
            # Count keyword occurrences to determine relevance
            keyword_count = sum(1 for keyword in self.f1_keywords if keyword in content_lower)
            # If many keywords are found, it's likely F1-related
            if keyword_count >= 3:
                return True
        
        return False
    
    async def extract_links(self, url: str) -> List[str]:
        """
        Extract links from a webpage.
        
        Args:
            url (str): URL to extract links from
            
        Returns:
            List[str]: List of extracted links
        """
        links = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                try:
                    await page.goto(url, timeout=30000)
                    # Get HTML content
                    html_content = await page.content()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Find all links
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = urljoin(url, href)
                        # Ensure it's a valid URL
                        if href.startswith(('http://', 'https://')):
                            links.append(href)
                    
                    # Check if content is F1 related before returning
                    text_content = soup.get_text(separator=' ', strip=True)
                    if self.is_f1_related(url, text_content):
                        self.f1_urls.append(url)
                        logger.info(f"✅ F1-related content found: {url}")
                    
                except TimeoutError:
                    logger.error(f"Timeout while loading {url}")
                finally:
                    await browser.close()
            
            return links
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return []
    
    async def crawl(self, start_urls: List[str]) -> List[str]:
        """
        Crawl F1-related websites starting from the provided URLs.
        
        Args:
            start_urls (List[str]): Starting URLs for crawling
            
        Returns:
            List[str]: List of discovered F1-related URLs
        """
        to_visit = start_urls.copy()
        current_depth = 0
        
        with Progress() as progress:
            task = progress.add_task("[green]Crawling F1 websites...", total=self.max_pages)
            
            while to_visit and len(self.visited_urls) < self.max_pages and current_depth <= self.depth:
                current_depth += 1
                next_level = []
                
                for url in to_visit:
                    if url in self.visited_urls:
                        continue
                    
                    self.visited_urls.add(url)
                    progress.update(task, advance=1, description=f"[green]Crawling: {url[:50]}...")
                    
                    links = await self.extract_links(url)
                    next_level.extend([link for link in links if link not in self.visited_urls])
                    
                    # Update progress
                    progress.update(task, completed=len(self.visited_urls), total=self.max_pages)
                    if len(self.visited_urls) >= self.max_pages:
                        break
                
                to_visit = next_level
                logger.info(f"Completed depth {current_depth}, discovered {len(self.f1_urls)} F1-related URLs")
        
        # Deduplicate and return results
        self.f1_urls = list(set(self.f1_urls))
        return self.f1_urls
    
    async def ingest_discovered_urls(self, max_chunks_per_url: int = 50) -> None:
        """
        Ingest discovered F1-related URLs into the RAG system.
        
        Args:
            max_chunks_per_url (int): Maximum chunks to extract per URL
        """
        if not self.f1_urls:
            logger.warning("No F1-related URLs to ingest. Run crawl() first.")
            return
        
        logger.info(f"Ingesting {len(self.f1_urls)} F1-related URLs into RAG system...")
        await self.f1_ai.ingest(self.f1_urls, max_chunks_per_url=max_chunks_per_url)
        logger.info("✅ Ingestion complete!")
    
    def save_urls_to_file(self, filename: str = "f1_urls.txt") -> None:
        """
        Save discovered F1 URLs to a text file.
        
        Args:
            filename (str): Name of the output file
        """
        if not self.f1_urls:
            logger.warning("No F1-related URLs to save. Run crawl() first.")
            return
        
        with open(filename, "w") as f:
            f.write(f"# F1-related URLs discovered on {datetime.now().isoformat()}\n")
            f.write(f"# Total URLs: {len(self.f1_urls)}\n\n")
            for url in self.f1_urls:
                f.write(f"{url}\n")
        
        logger.info(f"✅ Saved {len(self.f1_urls)} URLs to {filename}")

async def main():
    """Main function to run the F1 scraper."""
    parser = argparse.ArgumentParser(description="F1 Web Scraper to discover and ingest F1-related content")
    parser.add_argument("--start-urls", nargs="+", default=["https://www.formula1.com/"], 
                        help="Starting URLs for crawling")
    parser.add_argument("--max-pages", type=int, default=100,
                        help="Maximum number of pages to crawl")
    parser.add_argument("--depth", type=int, default=2,
                        help="Maximum crawl depth")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest discovered URLs into RAG system")
    parser.add_argument("--max-chunks", type=int, default=50,
                        help="Maximum chunks per URL for ingestion")
    parser.add_argument("--output", type=str, default="f1_urls.txt",
                        help="Output file for discovered URLs")
    parser.add_argument("--llm-provider", choices=["ollama", "openrouter"], default="openrouter",
                        help="Provider for LLM (default: openrouter)")
    
    args = parser.parse_args()
    
    # Initialize F1AI if needed
    f1_ai = None
    if args.ingest:
        f1_ai = F1AI(llm_provider=args.llm_provider)
    
    # Initialize and run the scraper
    scraper = F1Scraper(
        max_pages=args.max_pages,
        depth=args.depth,
        f1_ai=f1_ai
    )
    
    # Crawl to discover F1-related URLs
    console.print("[bold blue]Starting F1 web crawler[/bold blue]")
    discovered_urls = await scraper.crawl(args.start_urls)
    console.print(f"[bold green]Discovered {len(discovered_urls)} F1-related URLs[/bold green]")
    
    # Save URLs to file
    scraper.save_urls_to_file(args.output)
    
    # Ingest if requested
    if args.ingest:
        console.print("[bold yellow]Starting ingestion into RAG system...[/bold yellow]")
        await scraper.ingest_discovered_urls(max_chunks_per_url=args.max_chunks)
        console.print("[bold green]Ingestion complete![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())