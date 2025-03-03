import os
import uuid
import json
import time
import asyncio
import requests
import websockets
import gradio as gr
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000")
STATUS_REFRESH_MINUTES = 5
NEWS_CACHE_MINUTES = 10
SUPPORTED_STOCKS = ["NVDA", "TSLA", "GOOG", "AAPL", "META", "AMZN"]

# State management
conversation_states: Dict[str, dict] = {}
news_cache: Dict[str, dict] = {}
system_status: dict = {}
last_status_check: Optional[datetime] = None

# Theme configuration
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky"
).set(
    button_primary_background_fill="#2563eb",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    border_color_primary="#cbd5e1",
    block_background_fill="#f8fafc"
)

# Modified to return a regular dict, not a coroutine
def check_system_status(force: bool = False) -> dict:
    """Check and cache system status with rate limiting"""
    global last_status_check, system_status
    
    if not force and last_status_check and \
       (datetime.now() - last_status_check) < timedelta(minutes=STATUS_REFRESH_MINUTES):
        return system_status
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=5)
        response.raise_for_status()
        system_status = response.json()
        last_status_check = datetime.now()
        return system_status
    except Exception as e:
        return {"status": "error", "message": str(e)}

def format_status_html(status: dict) -> str:
    """Format system status information into HTML"""
    if status.get("status") == "running":
        return f"""
        <div class="status-box success">
            <h3>System Status: <span class="status-green">Operational</span></h3>
            <p>📈 Supported Stocks: {', '.join(status.get('supported_stocks', []))}</p>
            <p>🗞️ Total News Items: {status.get('total_news_items', 0)}</p>
            <p>📦 Vector Store Size: {status.get('vector_store_size', 0)}</p>
            <p>🕒 Last Update: {status.get('last_scrape_time', 'N/A')}</p>
        </div>
        """
    return f"""
    <div class="status-box error">
        <h3>System Status: <span class="status-red">Degraded</span></h3>
        <p>⚠️ {status.get('message', 'Unknown error')}</p>
    </div>
    """

async def websocket_chat_handler(question: str, conversation_id: str) -> str:
    """Handle WebSocket communication with the advisor"""
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    try:
        async with websockets.connect(f"{WS_BASE_URL}/ws/chat/{conversation_id}") as ws:
            await ws.send(question)
            response = await ws.recv()
            return response, conversation_id
    except Exception as e:
        return f"Connection error: {str(e)}", conversation_id

# Modified to make this function non-async and use regular requests
def get_stock_news(stock_symbol: str) -> str:
    """Retrieve and format stock news with caching"""
    stock_symbol = stock_symbol.upper()
    cached = news_cache.get(stock_symbol)
    
    if cached and (datetime.now() - cached["timestamp"]) < timedelta(minutes=NEWS_CACHE_MINUTES):
        return cached["html"]
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/news/{stock_symbol}", timeout=10)
        response.raise_for_status()
        news_data = response.json()
        
        html = f"""
        <div class="news-container">
            <h3>📰 Latest News for {stock_symbol}</h3>
            <div class="news-items">
        """
        
        for item in news_data["news"]:
            timestamp = datetime.fromisoformat(item["timestamp"].replace('Z', '+00:00'))
            html += f"""
            <div class="news-card">
                <h4>{item['title']}</h4>
                <p class="news-content">{item['content'][:200]}...</p>
                <div class="news-footer">
                    <span class="news-source">{item['source']}</span>
                    <span class="news-date">{timestamp.strftime('%b %d, %H:%M UTC')}</span>
                    <a href="{item['url']}" target="_blank" class="news-link">Read more →</a>
                </div>
            </div>
            """
        
        html += "</div></div>"
        news_cache[stock_symbol] = {"html": html, "timestamp": datetime.now()}
        return html
    except Exception as e:
        return f"❌ Error fetching news: {str(e)}"

def generate_mock_chart(stock_symbol: str) -> go.Figure:
    """Generate a mock stock chart for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    prices = np.random.normal(100, 15, 30).cumsum() + np.random.randint(100, 200)
    
    fig = go.Figure(data=[
        go.Scatter(
            x=dates,
            y=prices,
            line=dict(color='#2563eb', width=2),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        )
    ])
    
    fig.update_layout(
        title=f"{stock_symbol} Stock Price (Last 30 Days - Mock Data)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        margin=dict(t=40, b=20),
        hovermode="x unified"
    )
    return fig

def update_chat_history(conversation_id: str) -> str:
    """Format conversation history for display"""
    history = conversation_states.get(conversation_id, [])
    html = []
    
    for msg in history[-10:]:  # Show last 10 messages
        if msg["role"] == "user":
            html.append(f"""
            <div class="chat-message user">
                <div class="message-bubble user">{msg["content"]}</div>
                <div class="message-label">You</div>
            </div>
            """)
        else:
            html.append(f"""
            <div class="chat-message bot">
                <div class="message-bubble bot">{msg["content"]}</div>
                <div class="message-label">Advisor</div>
            </div>
            """)
    
    return f"<div class='chat-history'>{''.join(html)}</div>"

# Modified to handle async/sync behavior properly
async def handle_chat_submit(question: str, conversation_id: str):
    """Handle chat message submission"""
    if not question.strip():
        return "", conversation_id, update_chat_history(conversation_id)
    
    # Add user message to history
    if conversation_id not in conversation_states:
        conversation_states[conversation_id] = []
    conversation_states[conversation_id].append({"role": "user", "content": question})
    
    # Get advisor response
    response, new_conv_id = await websocket_chat_handler(question, conversation_id)
    conversation_id = new_conv_id
    
    # Add advisor response to history
    conversation_states[conversation_id].append({"role": "assistant", "content": response})
    
    return "", conversation_id, update_chat_history(conversation_id)

def create_interface():
    """Create the Gradio interface"""
    css = """
    .status-box { padding: 15px; border-radius: 8px; margin: 10px 0; }
    .status-green { color: #16a34a; }
    .status-red { color: #dc2626; }
    .chat-history { max-height: 400px; overflow-y: auto; padding: 10px; }
    .message-bubble { padding: 10px 15px; border-radius: 15px; margin: 5px 0; max-width: 80%; }
    .user .message-bubble { background: #dbeafe; margin-left: auto; border-radius: 15px 15px 0 15px; }
    .bot .message-bubble { background: #f1f5f9; margin-right: auto; border-radius: 15px 15px 15px 0; }
    .news-card { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    """
    
    with gr.Blocks(theme=THEME, css=css, title="Stock Trading Advisor") as app:
        # Header Section
        gr.Markdown("""
        # 📈 Stock Trading Advisor
        AI-powered financial insights for major tech stocks
        """)
        
        # System Status
        with gr.Row():
            status_display = gr.HTML()
            gr.Button("🔄 Refresh Status", size="sm").click(
                lambda: format_status_html(check_system_status(force=True)),
                outputs=status_display
            )
        
        # Main Tabs
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Chat Advisor"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chat_history = gr.HTML()
                        chat_input = gr.Textbox(placeholder="Ask about supported stocks...", label="Your Question")
                        chat_submit = gr.Button("Send", variant="primary")
                        conversation_id = gr.State()
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Example Questions")
                        gr.Examples(
                            examples=[
                                "Should I invest in NVDA right now?",
                                "Compare TSLA and GOOG as long-term investments",
                                "What's the latest news about META?",
                                "Analyze AAPL's recent stock performance"
                            ],
                            inputs=[chat_input]
                        )
            
            # Analysis Tab
            with gr.Tab("📊 Stock Analysis"):
                with gr.Row():
                    stock_selector = gr.Dropdown(
                        label="Select Stock",
                        choices=SUPPORTED_STOCKS,
                        value="NVDA"
                    )
                
                with gr.Row():
                    stock_chart = gr.Plot(label="Price Chart")
                    news_display = gr.HTML()
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This System
                **Stock Trading Advisor** is a RAG-powered financial analysis tool providing:
                - Real-time stock news aggregation
                - AI-powered investment insights
                - Technical analysis visualization
                
                ### Supported Stocks
                NVIDIA (NVDA), Tesla (TSLA), Alphabet (GOOG), Apple (AAPL), Meta (META), Amazon (AMZN)
                
                ### Disclaimer
                This system provides informational purposes only. Never consider this as financial advice.
                Always consult a qualified professional before making investment decisions.
                """)
        
        # Event Handling
        chat_submit.click(
            handle_chat_submit,
            [chat_input, conversation_id],
            [chat_input, conversation_id, chat_history]
        )
        
        stock_selector.change(
            lambda s: (generate_mock_chart(s), get_stock_news(s)),
            [stock_selector],
            [stock_chart, news_display]
        )
        
        # Initialization - Fixed to use synchronous function
        app.load(
            lambda: format_status_html(check_system_status()),
            outputs=status_display
        )
        stock_selector.change(None, stock_selector)
    
    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        favicon_path=None
    )