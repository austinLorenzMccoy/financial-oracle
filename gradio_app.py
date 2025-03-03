import os
import uuid
import requests
import gradio as gr
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SUPPORTED_STOCKS = ["NVDA", "TSLA", "GOOG", "AAPL", "META", "AMZN"]
NEWS_CACHE_MINUTES = 10

# Cache
news_cache = {}


def get_stock_news(stock_symbol: str):
    """Retrieve stock news with caching."""
    stock_symbol = stock_symbol.upper()
    cached = news_cache.get(stock_symbol)
    
    if cached and (datetime.now() - cached["timestamp"]) < timedelta(minutes=NEWS_CACHE_MINUTES):
        return cached["news"]
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/news/{stock_symbol}", timeout=5)
        response.raise_for_status()
        news_data = response.json()
        news_cache[stock_symbol] = {"news": news_data, "timestamp": datetime.now()}
        return news_data
    except Exception as e:
        return {"error": str(e)}


def ask_advisor(question: str, conversation_id: str = None):
    """Ask the financial advisor a question."""
    conversation_id = conversation_id or str(uuid.uuid4())
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/ask",
            json={"question": question, "conversation_id": conversation_id},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data["response"], data["conversation_id"]
    except Exception as e:
        return f"Error: {str(e)}", conversation_id


def create_interface():
    """Builds the Gradio UI."""
    with gr.Blocks(title="Stock Trading Advisor") as app:
        gr.Markdown("""# 📈 Stock Trading Advisor
        Get AI-powered financial insights on major stocks.""")
        
        with gr.Row():
            chat_input = gr.Textbox(placeholder="Ask about stocks...", label="Your Question")
            chat_submit = gr.Button("Ask", variant="primary")
        
        chat_output = gr.Textbox(label="Advisor's Response", interactive=False)
        conversation_id = gr.State()
        
        chat_submit.click(ask_advisor, inputs=[chat_input, conversation_id], outputs=[chat_output, conversation_id])
        
        with gr.Row():
            stock_selector = gr.Dropdown(label="Select Stock", choices=SUPPORTED_STOCKS, value="NVDA")
            news_display = gr.Textbox(label="Latest News", interactive=False, lines=5)
        
        stock_selector.change(lambda s: get_stock_news(s), inputs=[stock_selector], outputs=[news_display])
        
    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
