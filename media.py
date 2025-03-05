import os
import sys
import asyncio
import logging
import nltk

# Data Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Natural Language Processing
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Project Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import get_financial_advice_system, FinancialAdviceSystem

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_media.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources
nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    """Comprehensive sentiment analysis utility."""
    
    @staticmethod
    def analyze(text: str) -> dict:
        try:
            blob = TextBlob(text)
            sia = SentimentIntensityAnalyzer()
            
            return {
                'textblob': {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                },
                'nltk': sia.polarity_scores(text),
                'composite_score': (
                    0.5 * blob.sentiment.polarity +
                    0.5 * sia.polarity_scores(text)['compound']
                )
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                'textblob': {'polarity': 0, 'subjectivity': 0},
                'nltk': {'compound': 0},
                'composite_score': 0
            }


class NewsAnalyzer:
    """Advanced news analysis for stocks."""
    
    def __init__(self, financial_system: FinancialAdviceSystem):
        self.system = financial_system
    
    async def analyze_stock_news(self, stock_symbol: str) -> dict:
        try:
            # Use synchronous method to get news
            news_items = self.system.get_stock_news(stock_symbol)
            
            if not news_items:
                return {'error': f'No news available for {stock_symbol}'}
            
            sentiments = [SentimentAnalyzer.analyze(item['title']) for item in news_items]
            composite_scores = [s['composite_score'] for s in sentiments]
            
            return {
                'stock': stock_symbol,
                'total_news_items': len(news_items),
                'sentiment_summary': {
                    'avg_polarity': np.mean(composite_scores) if composite_scores else 0,
                    'sentiment_distribution': self._sentiment_distribution(sentiments)
                },
                'news_samples': [
                    {'title': item['title'], 'sentiment': sent['composite_score']}
                    for item, sent in zip(news_items[:3], sentiments[:3])
                ]
            }
        except Exception as e:
            logger.error(f"News analysis error for {stock_symbol}: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def _sentiment_distribution(sentiments: list) -> dict:
        dist = {'very_positive': 0, 'positive': 0, 'neutral': 0, 'negative': 0, 'very_negative': 0}
        for s in sentiments:
            score = s['composite_score']
            if score > 0.5:
                dist['very_positive'] += 1
            elif score > 0:
                dist['positive'] += 1
            elif score < -0.5:
                dist['very_negative'] += 1
            elif score < 0:
                dist['negative'] += 1
            else:
                dist['neutral'] += 1
        return dist


class StockVisualizer:
    """Visualization utilities for stock analysis."""
    
    @staticmethod
    def plot_sentiment(analysis: dict, filename: str = 'sentiment.png') -> str:
        try:
            plt.figure(figsize=(10, 6))
            dist = analysis['sentiment_summary']['sentiment_distribution']
            plt.bar(dist.keys(), dist.values(), color=['green', 'lightgreen', 'gray', 'salmon', 'red'])
            plt.title(f"{analysis['stock']} News Sentiment")
            plt.xlabel("Sentiment Category")
            plt.ylabel("Number of Articles")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            return filename
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return ''


async def main():
    """Demonstrate news analysis with proper async handling."""
    try:
        # Initialize the system
        system = get_financial_advice_system()
        
        # Start the system to ensure initial data collection
        await system.start()
        
        # Allow time for initial scraping
        logger.info("Waiting for initial data collection...")
        await asyncio.sleep(30)  # Extended warm-up period
        
        # Create analyzer and visualizer
        analyzer = NewsAnalyzer(system)
        visualizer = StockVisualizer()
        
        # Stocks to analyze
        stocks = ['NVDA', 'TSLA', 'GOOG', 'AAPL', 'META', 'AMZN']
        
        # Analyze each stock
        for symbol in stocks:
            logger.info(f"Analyzing {symbol}...")
            try:
                # Perform analysis
                analysis = await analyzer.analyze_stock_news(symbol)
                
                # Check for errors
                if 'error' in analysis:
                    logger.warning(f"Analysis failed for {symbol}: {analysis['error']}")
                    continue
                
                # Display results
                print(f"\n{symbol} Analysis:")
                print(f"Total Articles: {analysis['total_news_items']}")
                print(f"Average Sentiment: {analysis['sentiment_summary']['avg_polarity']:.2f}")
                
                # Generate and show visualization
                plot_path = visualizer.plot_sentiment(analysis, f"{symbol}_sentiment.png")
                if plot_path:
                    print(f"Sentiment visualization saved to {plot_path}")
                
                # Show top news samples
                print("\nTop News Headlines:")
                for idx, news in enumerate(analysis['news_samples'], 1):
                    print(f"{idx}. {news['title']} (Sentiment: {news['sentiment']:.2f})")
                    
            except Exception as sym_error:
                logger.error(f"Error processing {symbol}: {sym_error}")
                
    except Exception as e:
        logger.critical(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted")
    except Exception as e:
        logger.error(f"Critical failure: {e}")