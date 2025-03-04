import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import logging
import asyncio
import async_timeout
import schedule
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

import pytz
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# Constants
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TEMPERATURE = 0.5
SCRAPE_INTERVAL_HOURS = 6
DATA_RETENTION_HOURS = 24
SUPPORTED_STOCKS = {"NVDA", "TSLA", "GOOG", "AAPL", "META", "AMZN"}
COMPANY_NAME_MAP = {
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "ALPHABET": "GOOG",
    "GOOGLE": "GOOG",
    "APPLE": "AAPL",
    "META": "META",
    "FACEBOOK": "META",
    "AMAZON": "AMZN"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_advisor.log')
    ]
)
logger = logging.getLogger("stock_advisor")


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))

    @field_validator('content')
    @classmethod
    def clean_content(cls, v):
        """Remove any role prefixes from the content."""
        v = v.strip()
        prefixes_to_remove = [
            "MessageRole.ASSISTANT:", "MessageRole.ASSISTANT",
            "Assistant:", "assistant:", "ASSISTANT:",
            "user:", "USER:", "system:", "SYSTEM:",
            "Response:", "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if v.startswith(prefix):
                v = v[len(prefix):].strip()
                break
        return v


class ConversationState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    last_activity: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))
    active: bool = True


class StockNewsItem(BaseModel):
    stock_symbol: str
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))


class NewsRepository:
    """Repository for managing stock news items."""
    
    def __init__(self):
        self.news_data: List[StockNewsItem] = []
        self._lock = asyncio.Lock()
        
    async def add_news(self, news_items: List[StockNewsItem]):
        """Add news items to the repository."""
        async with self._lock:
            self.news_data.extend(news_items)
            logger.info(f"Added {len(news_items)} news items to repository")
    
    async def get_news_for_stock(self, stock_symbol: str) -> List[StockNewsItem]:
        """Get news items for a specific stock."""
        async with self._lock:
            return [item for item in self.news_data if item.stock_symbol == stock_symbol]
    
    async def get_all_news(self) -> List[StockNewsItem]:
        """Get all news items."""
        async with self._lock:
            return self.news_data.copy()
    
    async def clean_old_news(self):
        """Remove news items older than DATA_RETENTION_HOURS."""
        current_time = datetime.now(pytz.UTC)
        cutoff_time = current_time - timedelta(hours=DATA_RETENTION_HOURS)
        
        async with self._lock:
            initial_count = len(self.news_data)
            self.news_data = [item for item in self.news_data 
                             if item.timestamp > cutoff_time]
            removed_count = initial_count - len(self.news_data)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} news items older than {DATA_RETENTION_HOURS} hours")


class WebScraper:
    """Scraper for financial news websites."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def scrape_news_for_stock(self, stock_symbol: str) -> List[StockNewsItem]:
        """Scrape news for a specific stock from multiple sources."""
        if stock_symbol not in SUPPORTED_STOCKS:
            logger.warning(f"Unsupported stock symbol: {stock_symbol}")
            return []
            
        try:
            logger.info(f"Scraping news for {stock_symbol}")
            news_items = []
            
            # Run scrapers concurrently
            yahoo_news_task = asyncio.create_task(self._scrape_yahoo_finance(stock_symbol))
            
            # Wait for Yahoo News first
            yahoo_news = await yahoo_news_task
            news_items.extend(yahoo_news)
            
            # Only try MarketWatch if Yahoo didn't return enough news
            if len(news_items) < 3:
                try:
                    marketwatch_news = await self._scrape_marketwatch(stock_symbol)
                    news_items.extend(marketwatch_news)
                except Exception as e:
                    logger.warning(f"Error scraping MarketWatch for {stock_symbol}: {str(e)}")
            
            logger.info(f"Successfully scraped {len(news_items)} news items for {stock_symbol}")
            return news_items
        except Exception as e:
            logger.error(f"Error scraping news for {stock_symbol}: {str(e)}")
            return []
    
    async def _scrape_yahoo_finance(self, stock_symbol: str) -> List[StockNewsItem]:
        """Scrape Yahoo Finance for news."""
        try:
            url = f"https://finance.yahoo.com/quote/{stock_symbol}/news"
            response = await self._fetch_url(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response, 'html.parser')
            news_items = []
            
            # Find news articles
            article_tags = soup.find_all('div', {'class': 'Ov(h)'})
            
            for article in article_tags[:10]:  # Limit to 10 articles
                try:
                    title_tag = article.find('h3')
                    if not title_tag:
                        continue
                        
                    title = title_tag.text.strip()
                    link_tag = article.find('a')
                    if not link_tag:
                        continue
                        
                    article_url = "https://finance.yahoo.com" + link_tag['href'] if link_tag['href'].startswith('/') else link_tag['href']
                    
                    # Get article content (summary)
                    summary_tag = article.find('p')
                    content = summary_tag.text.strip() if summary_tag else "No summary available"
                    
                    news_items.append(StockNewsItem(
                        stock_symbol=stock_symbol,
                        title=title,
                        content=content,
                        source="Yahoo Finance",
                        url=article_url
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing Yahoo Finance article: {str(e)}")
            
            return news_items
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance for {stock_symbol}: {str(e)}")
            return []
    
    async def _scrape_marketwatch(self, stock_symbol: str) -> List[StockNewsItem]:
        """Scrape MarketWatch for news."""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{stock_symbol.lower()}"
            response = await self._fetch_url(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response, 'html.parser')
            news_items = []
            
            # Find news articles
            article_tags = soup.find_all('div', {'class': 'article__content'})
            
            for article in article_tags[:5]:  # Limit to 5 articles
                try:
                    title_tag = article.find('h3', {'class': 'article__headline'})
                    if not title_tag:
                        continue
                        
                    title = title_tag.text.strip()
                    link_tag = title_tag.find('a')
                    if not link_tag:
                        continue
                        
                    article_url = link_tag['href']
                    
                    # Get article summary
                    summary_tag = article.find('p', {'class': 'article__summary'})
                    content = summary_tag.text.strip() if summary_tag else "No summary available"
                    
                    news_items.append(StockNewsItem(
                        stock_symbol=stock_symbol,
                        title=title,
                        content=content,
                        source="MarketWatch",
                        url=article_url
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing MarketWatch article: {str(e)}")
            
            return news_items
        except Exception as e:
            logger.error(f"Error scraping MarketWatch for {stock_symbol}: {str(e)}")
            return []
            
    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content with error handling and retries using aiohttp."""
        max_retries = 3
        retry_delay = 2  # seconds
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(headers=self.headers, timeout=timeout) as session:
            for attempt in range(max_retries):
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.text()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error fetching {url}: {str(e)} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
            return None
            
    def _make_request(self, url: str) -> str:
        """Make an HTTP request using requests library."""
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.text


class StockAdvisorRAG:
    """RAG-based stock advisor system."""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        self.vectorstore = None
        self.retrieval_chain = None
        
        # Initialize with placeholder
        self._initialize_empty_vectorstore()
        
    def _initialize_empty_vectorstore(self):
        """Initialize an empty vector store."""
        try:
            # Create with placeholder document
            self.vectorstore = FAISS.from_texts(
                ["Financial advice system initialized. Ready to provide information on stocks."], 
                self.embeddings
            )
            self.retrieval_chain = self._create_rag_chain()
            logger.info("Initialized empty vector store")
        except Exception as e:
            logger.error(f"Failed to initialize empty vector store: {str(e)}")
            raise
    
    def _create_rag_chain(self):
        """Create the RAG chain for stock advice."""
        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert financial advisor specializing in stock analysis for 
            NVIDIA (NVDA), Tesla (TSLA), Alphabet (GOOG), Apple (AAPL), Meta (META), and Amazon (AMZN).
            
            Use the following context information to provide investment advice:
            {context}
            
            Current date: {current_date}
            
            Remember to:
            1. Clearly explain the advantages and disadvantages of buying the stock
            2. Consider recent news and market trends from the context
            3. Be balanced in your assessment, mentioning both positive and negative factors
            4. Avoid making definitive predictions about stock performance
            5. Include a disclaimer that this is not professional financial advice
            
            Be direct and concise in your response.
            """),
            ("human", "{input}")
        ])
    
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retriever
        retriever = self.vectorstore.as_retriever()
        
        # Define format function to properly structure inputs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build the RAG chain with the correct structure
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough(), "current_date": RunnablePassthrough()}
            | document_chain
        )
        
        return rag_chain
        
    async def update_vectorstore(self, news_items: List[StockNewsItem]):
        """Update the vector store with news items."""
        if not news_items:
            logger.info("No news items to update vector store")
            return
            
        try:
            logger.info(f"Updating vector store with {len(news_items)} news items")
            
            # Prepare text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            texts = []
            for news in news_items:
                formatted_text = (
                    f"Stock: {news.stock_symbol}\n"
                    f"Title: {news.title}\n"
                    f"Source: {news.source}\n"
                    f"Date: {news.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    f"Content: {news.content}\n"
                    f"URL: {news.url}\n"
                )
                chunks = text_splitter.split_text(formatted_text)
                texts.extend(chunks)
            
            # Run in a separate thread to avoid blocking
            if self.vectorstore:
                # Add to existing vectorstore
                await asyncio.to_thread(self.vectorstore.add_texts, texts)
                logger.info(f"Added {len(texts)} chunks to existing vector store")
            else:
                # Create new vectorstore
                self.vectorstore = await asyncio.to_thread(
                    FAISS.from_texts, texts, self.embeddings
                )
                logger.info(f"Created new vector store with {len(texts)} chunks")
            
            # Update retrieval chain
            self.retrieval_chain = self._create_rag_chain()
            logger.info("Vector store updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
    
    async def generate_response(self, query: str, conversation_context: str = "") -> str:
        """Generate a response to a stock-related query."""
        try:
            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Add context to query if available
            if conversation_context:
                enhanced_query = f"Previous conversation context:\n{conversation_context}\n\nCurrent question: {query}"
            else:
                enhanced_query = query
                
            # Try using the RAG chain
            try:
                if self.retrieval_chain:
                    # Process with RAG
                    response = await asyncio.to_thread(
                        self.retrieval_chain.invoke,
                        {"input": enhanced_query, "current_date": current_date}
                    )
                    return response.content  # Extract content from AIMessage
                else:
                    raise ValueError("Retrieval chain not initialized")
                    
            except Exception as chain_error:
                logger.error(f"Error using retrieval chain: {str(chain_error)}")
                
                # Fallback to direct LLM call
                fallback_prompt = f"""
                I'm a financial advisor specializing in stock analysis. A user has asked the following question:
                
                {query}
                
                Today's date is {current_date}.
                
                Please provide a balanced response about the stock(s) mentioned, or explain that I can only provide advice on 
                NVIDIA (NVDA), Tesla (TSLA), Alphabet (GOOG), Apple (AAPL), Meta (META), and Amazon (AMZN) stocks.
                
                Include advantages and disadvantages of buying the stock based on recent market trends.
                Add a disclaimer that this is not professional financial advice.
                """
                
                response = await asyncio.to_thread(
                    self.llm.invoke,
                    fallback_prompt
                )
                return response.content.strip()
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request. Please try again later."


class FinancialAdviceSystem:
    """Main class for the financial advice system."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
            
        # Initialize components
        self.news_repository = NewsRepository()
        self.web_scraper = WebScraper()
        self.rag_system = StockAdvisorRAG(groq_api_key)
        self.conversation_storage: Dict[str, ConversationState] = {}
        
        # Background tasks
        self.background_tasks = set()
        
    async def start(self):  # Make this async
        """Start the financial advice system."""
        logger.info("Starting Financial Advice System")
        
        # Start initial data scraping
        self._create_background_task(self.scrape_all_stocks())
        
        # Schedule periodic tasks
        self._create_background_task(self._schedule_tasks())
        
    async def _schedule_tasks(self):
        """Schedule and run periodic tasks."""
        while True:
            # Schedule scraping every SCRAPE_INTERVAL_HOURS
            self._create_background_task(self.scrape_all_stocks())
            
            # Schedule cleanup every hour
            self._create_background_task(self.news_repository.clean_old_news())
            
            # Wait for the next interval
            await asyncio.sleep(SCRAPE_INTERVAL_HOURS * 3600)
            
    def _create_background_task(self, coro):
        """Create and track a background task."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def scrape_all_stocks(self):
        """Scrape news for all supported stocks with enhanced error handling."""
        logger.info("Starting news scraping for all stocks")
        all_news = []
        
        # Create a task for each stock
        tasks = [self.web_scraper.scrape_news_for_stock(stock) for stock in SUPPORTED_STOCKS]
        
        # Run with some concurrency control (3 at a time)
        for i in range(0, len(tasks), 3):
            batch_results = await asyncio.gather(*tasks[i:i+3], return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                stock_idx = i + j
                stock = list(SUPPORTED_STOCKS)[stock_idx] if stock_idx < len(SUPPORTED_STOCKS) else "Unknown"
                
                if isinstance(result, Exception):
                    logger.error(f"Error scraping {stock}: {str(result)}")
                    # Add a fallback news item if scraping fails
                    all_news.append(StockNewsItem(
                        stock_symbol=stock,
                        title=f"Latest market trends for {stock}",
                        content=f"This is a placeholder for {stock} news. The system was unable to scrape real-time news at this moment.",
                        source="System Fallback",
                        url=f"https://finance.yahoo.com/quote/{stock}"
                    ))
                elif isinstance(result, list):
                    if result:
                        all_news.extend(result)
                        logger.info(f"Successfully scraped {len(result)} news items for {stock}")
                    else:
                        logger.warning(f"No news found for {stock}, adding fallback")
                        # Add a fallback news item if no news found
                        all_news.append(StockNewsItem(
                            stock_symbol=stock,
                            title=f"Latest market trends for {stock}",
                            content=f"No recent news was found for {stock}. Consider checking major financial news sources for the latest updates.",
                            source="System Notice",
                            url=f"https://finance.yahoo.com/quote/{stock}"
                        ))
            
            # Brief pause between batches
            await asyncio.sleep(2)
        
        if all_news:
            await self.news_repository.add_news(all_news)
            await self.rag_system.update_vectorstore(all_news)
            logger.info(f"Completed scraping with {len(all_news)} total news items")
        else:
            logger.warning("No news items found during scraping. Adding fallback data.")
            # Add fallback data for all stocks if no news was found at all
            fallback_news = []
            for stock in SUPPORTED_STOCKS:
                fallback_news.append(StockNewsItem(
                    stock_symbol=stock,
                    title=f"Market overview for {stock}",
                    content=f"The system is currently unable to retrieve latest news for {stock}. Please try again later or check major financial news sources.",
                    source="System Fallback",
                    url=f"https://finance.yahoo.com/quote/{stock}"
                ))
            await self.news_repository.add_news(fallback_news)
            await self.rag_system.update_vectorstore(fallback_news)
            logger.info(f"Added {len(fallback_news)} fallback news items")
    
    def extract_stock_symbols(self, query: str) -> Set[str]:
        """Extract mentioned stock symbols from the query."""
        mentioned_stocks = set()
        
        # Check for stock symbols
        for stock in SUPPORTED_STOCKS:
            if stock in query.upper():
                mentioned_stocks.add(stock)
        
        # Check for company names
        for company_name, symbol in COMPANY_NAME_MAP.items():
            if company_name in query.upper() or company_name.capitalize() in query:
                mentioned_stocks.add(symbol)
        
        return mentioned_stocks
    
    def get_conversation_context(self, conversation_id: str, context_window: int = 5) -> str:
        """Get recent conversation context by ID."""
        if conversation_id not in self.conversation_storage:
            return ""
        
        conversation = self.conversation_storage[conversation_id]
        recent_messages = conversation.messages[-context_window:]
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
    
    def add_message_to_conversation(self, conversation_id: str, role: MessageRole, content: str):
        """Add a message to a conversation."""
        if conversation_id not in self.conversation_storage:
            self.conversation_storage[conversation_id] = ConversationState()
        
        conversation = self.conversation_storage[conversation_id]
        message = Message(role=role, content=content)
        conversation.messages.append(message)
        conversation.last_activity = datetime.now(pytz.UTC)
    
    async def generate_response(self, question: str, conversation_id: str = None) -> str:
        """Generate a response to a user question."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        try:
            # Add user message to conversation
            self.add_message_to_conversation(conversation_id, MessageRole.USER, question)
            
            # Get conversation context
            context = self.get_conversation_context(conversation_id)
            
            # Generate response
            response = await self.rag_system.generate_response(question, context)
            
            # Add assistant response to conversation
            self.add_message_to_conversation(conversation_id, MessageRole.ASSISTANT, response)
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            error_msg = "I'm sorry, I encountered an error while processing your request. Please try again later."
            self.add_message_to_conversation(conversation_id, MessageRole.ASSISTANT, error_msg)
            return error_msg
    
    def get_stock_news(self, stock_symbol: str) -> List[dict]:
        """Get news for a specific stock symbol."""
        if stock_symbol not in SUPPORTED_STOCKS:
            return []
        
        # Convert to sync version for API compatibility
        news_items = []
        for item in self.news_repository.news_data:
            if item.stock_symbol == stock_symbol:
                news_items.append({
                    "title": item.title,
                    "content": item.content,
                    "source": item.source,
                    "url": item.url,
                    "timestamp": item.timestamp.isoformat()
                })
        
        return news_items


# Global instance
_financial_advice_system = None

def get_financial_advice_system():
    """Get or create the global financial advice system instance."""
    global _financial_advice_system
    if _financial_advice_system is None:
        _financial_advice_system = FinancialAdviceSystem()
    return _financial_advice_system


async def main():
    """Main entry point for testing."""
    # Initialize the system
    system = get_financial_advice_system()
    
    # Wait for initial data scraping
    print("Waiting for initial data scraping...")
    await asyncio.sleep(15)
    
    # Test queries
    test_queries = [
        "Should I buy NVDA stock?",
        "What do you think about Tesla's stock prospects?",
        "Compare Google and Apple stocks",
        "What are the latest news for Meta stock?"
    ]
    
    conversation_id = str(uuid.uuid4())
    
    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("-" * 50)
        response = await system.generate_response(query, conversation_id)
        print(f"Response: {response}")
        print("=" * 50)
        # Brief pause between queries
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())