import os
import uuid
import requests
import gradio as gr
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SUPPORTED_STOCKS = ["NVDA", "TSLA", "GOOG", "AAPL", "META", "AMZN"]
REFRESH_INTERVAL = 300  # 5 minutes

# Cache
cache = {
    "news": {},
    "sentiment": {},
    "plots": {}
}

def get_stock_news(stock_symbol: str):
    """Retrieve stock news with caching"""
    stock_symbol = stock_symbol.upper()
    cached = cache["news"].get(stock_symbol)
    
    if cached and (datetime.now() - cached["timestamp"]) < timedelta(minutes=10):
        return cached["data"]
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/news/{stock_symbol}", timeout=10)
        response.raise_for_status()
        news_data = response.json()
        cache["news"][stock_symbol] = {"data": news_data, "timestamp": datetime.now()}
        return news_data
    except Exception as e:
        return {"error": str(e)}

def get_sentiment_analysis(stock_symbol: str):
    """Get sentiment analysis with caching"""
    stock_symbol = stock_symbol.upper()
    cached = cache["sentiment"].get(stock_symbol)
    
    if cached and (datetime.now() - cached["timestamp"]) < timedelta(seconds=REFRESH_INTERVAL):
        return cached["data"]
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/sentiment/{stock_symbol}", timeout=10)
        response.raise_for_status()
        sentiment_data = response.json()
        cache["sentiment"][stock_symbol] = {"data": sentiment_data, "timestamp": datetime.now()}
        return sentiment_data
    except Exception as e:
        return {"error": str(e)}

def get_sentiment_plot(stock_symbol: str):
    """Get sentiment visualization with caching"""
    stock_symbol = stock_symbol.upper()
    cached = cache["plots"].get(stock_symbol)
    
    if cached and (datetime.now() - cached["timestamp"]) < timedelta(seconds=REFRESH_INTERVAL):
        return cached["data"]
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/sentiment/plot/{stock_symbol}", timeout=10)
        response.raise_for_status()
        plot_data = BytesIO(response.content)
        cache["plots"][stock_symbol] = {"data": plot_data, "timestamp": datetime.now()}
        return plot_data
    except Exception as e:
        return None

def format_news(news_data: dict) -> str:
    """Format news items for display"""
    if "error" in news_data:
        return f"❌ Error: {news_data['error']}"
    
    return "\n\n".join(
        f"📰 {item['title']}\n{item['content'][:100]}..."
        for item in news_data.get("news", [])[:3]
    )

def format_sentiment(sentiment_data: dict) -> str:
    """Format sentiment analysis results"""
    if "error" in sentiment_data:
        return f"❌ Error: {sentiment_data['error']}"
    
    return (
        f"📊 Sentiment Analysis\n"
        f"Average Polarity: {sentiment_data.get('average_polarity', 0):.2f}\n"
        f"Positive: {sentiment_data.get('sentiment_distribution', {}).get('positive', 0)}\n"
        f"Negative: {sentiment_data.get('sentiment_distribution', {}).get('negative', 0)}\n"
        f"Neutral: {sentiment_data.get('sentiment_distribution', {}).get('neutral', 0)}"
    )

def update_stock_analysis(stock_symbol: str):
    """Update all stock-related components"""
    news = get_stock_news(stock_symbol)
    sentiment = get_sentiment_analysis(stock_symbol)
    plot = get_sentiment_plot(stock_symbol)
    
    return (
        format_news(news),
        format_sentiment(sentiment),
        plot if plot else "sentiment_plot.png"
    )

def ask_advisor(question: str, conversation_id: str = None):
    """Ask the financial advisor a question"""
    conversation_id = conversation_id or str(uuid.uuid4())
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/ask",
            json={"question": question, "conversation_id": conversation_id},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        return data["response"], data["conversation_id"]
    except Exception as e:
        return f"Error: {str(e)}", conversation_id

def create_interface():
    """Build the Gradio interface with media analysis"""
    css = """
    .analysis-box { padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e5e7eb; }
    .news-item { margin: 10px 0; padding: 10px; background: #f8fafc; border-radius: 8px; }
    """
    
    with gr.Blocks(title="Stock Advisor", css=css) as app:
        # Chat Section
        gr.Markdown("# 📈 AI Stock Trading Advisor")
        
        with gr.Row():
            with gr.Column(scale=2):
                chat_history = gr.Textbox(label="Conversation History", interactive=False, lines=8)
                chat_input = gr.Textbox(placeholder="Ask about stocks...", label="Your Question")
                chat_submit = gr.Button("Ask", variant="primary")
                conversation_id = gr.State()
                
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Stock Analysis")
                stock_selector = gr.Dropdown(
                    label="Select Stock",
                    choices=SUPPORTED_STOCKS,
                    value="NVDA"
                )
                sentiment_plot = gr.Image(label="Sentiment Analysis", interactive=False)
                sentiment_display = gr.Textbox(label="Sentiment Summary", interactive=False)
                news_display = gr.Textbox(label="Latest News", interactive=False, lines=6)
        
        # Event Handlers
        chat_submit.click(
            fn=ask_advisor,
            inputs=[chat_input, conversation_id],
            outputs=[chat_history, conversation_id]
        )
        
        stock_selector.change(
            fn=update_stock_analysis,
            inputs=stock_selector,
            outputs=[news_display, sentiment_display, sentiment_plot]
        )
        
        # Initial load
        app.load(
            fn=lambda: update_stock_analysis("NVDA"),
            outputs=[news_display, sentiment_display, sentiment_plot]
        )
    
    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )