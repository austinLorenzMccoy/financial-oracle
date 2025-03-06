import os
import uuid
import base64
import json
import gradio as gr
import requests
import websockets
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000")
SUPPORTED_STOCKS = ["NVDA", "TSLA", "GOOG", "AAPL", "META", "AMZN"]

# State management
conversation_states: Dict[str, dict] = {}
media_cache: Dict[str, dict] = {}

# Custom CSS
CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.chat-message {
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
    max-width: 80%;
}
.user-message {
    background: #e3f2fd;
    margin-left: auto;
}
.bot-message {
    background: #f5f5f5;
}
.media-card {
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 12px 0;
}
.status-indicator {
    padding: 8px 12px;
    border-radius: 20px;
    font-weight: 500;
}
.status-green {
    background: #e8f5e9;
    color: #2e7d32;
}
.status-red {
    background: #ffebee;
    color: #c62828;
}
"""

# Theme configuration
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal"
).set(
    button_primary_background_fill_hover="#1976d2",
    button_secondary_background_fill="#e0e0e0",
    button_secondary_background_fill_hover="#bdbdbd",
    body_text_color="#212121",
    background_fill_primary="#fafafa",
    border_color_primary="#e0e0e0"
)

@lru_cache(maxsize=32)
def cached_api_call(endpoint: str, params: dict = None):
    """Cached API calls for static data"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

async def websocket_chat_handler(question: str, conversation_id: str, generate_audio: bool):
    """Handle WebSocket communication with typing indicator"""
    async with websockets.connect(f"{WS_BASE_URL}/ws/chat/{conversation_id}") as ws:
        await ws.send(json.dumps({
            "type": "text",
            "message": question,
            "generate_audio": generate_audio
        }))
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "typing":
                yield "", conversation_id, "", True, None
            elif data["type"] == "message":
                audio = None
                if "base64_audio" in data:
                    audio = (data["sample_rate"], np.frombuffer(base64.b64decode(data["base64_audio"]), dtype=np.float32))
                yield data["message"], conversation_id, "", False, audio
                break
            elif data["type"] == "error":
                yield data["message"], conversation_id, "⚠️ " + data["message"], False, None
                break

def create_sentiment_plot(stock: str):
    """Create interactive sentiment analysis plot"""
    data = cached_api_call(f"api/sentiment/{stock}")
    if "error" in data:
        return None
    
    df = pd.DataFrame({
        'sentiment': list(data['sentiment_distribution'].keys()),
        'count': list(data['sentiment_distribution'].values())
    })
    
    fig = px.pie(df, values='count', names='sentiment', 
                title=f"{stock} Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def update_media_display(stock: str):
    """Update media section with latest analysis"""
    media = cached_api_call(f"api/media-analysis", {
        "stock_symbol": stock,
        "generate_plots": True,
        "generate_videos": True
    })
    
    outputs = []
    for item in media.get("media_files", []):
        if item["mime_type"] == "image/png":
            outputs.append(gr.Image(value=item["path"], label="Sentiment Analysis"))
        elif item["mime_type"] == "video/mp4":
            outputs.append(gr.Video(value=item["path"], label="Market Analysis Video"))
    return outputs

def create_voice_interface():
    """Create voice interaction components"""
    with gr.Row():
        with gr.Column(scale=1):
            record_button = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
        with gr.Column(scale=4):
            voice_output = gr.Audio(autoplay=True, label="Advisor Response")
    return record_button, voice_output

def create_chat_interface():
    """Create main chat interface"""
    with gr.Row():
        with gr.Column(scale=3):
            chat_history = gr.Chatbot(
                label="Conversation History",
                avatar_images=("user.png", "bot.png"),
                height=600,
                type="messages"  # Fixing the deprecation warning
            )
            chat_input = gr.Textbox(
                placeholder="Ask about stock investments or say 'help' for options...",
                label="Your Message"
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                voice_btn = gr.Button("🎙️ Voice Input")
                audio_toggle = gr.Checkbox(label="Enable Voice Responses")
            voice_output = gr.Audio(autoplay=True, label="Audio Response", visible=False)
        with gr.Column(scale=1):
            gr.Markdown("### Quick Analysis Tools")
            stock_selector = gr.Dropdown(SUPPORTED_STOCKS, label="Select Stock", value="NVDA")
            sentiment_plot = gr.Plot(label="Market Sentiment")
            media_display = gr.Tabs([
                gr.Tab("📈 Charts", [gr.Image(label="Analysis Chart")]),
                gr.Tab("🎥 Videos", [gr.Video(label="Analysis Video")])
            ])
    
    return (chat_history, chat_input, submit_btn, voice_btn, 
           audio_toggle, stock_selector, sentiment_plot, media_display, voice_output)

def create_system_status():
    """Create system status components"""
    with gr.Row():
        with gr.Column():
            status_indicator = gr.Markdown("### System Status: <span class='status-indicator status-green'>● Operational</span>")
            gr.Markdown("""
            **Last Updated**: Loading...  
            **News Items**: 0  
            **Active Conversations**: 0
            """)
        gr.Button("🔄 Refresh", variant="secondary")

def handle_voice_input(audio_path: str):
    """Process voice input through speech recognition"""
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                f"{API_BASE_URL}/api/speech-recognition",
                files={"file": audio_file},
                data={"save_recording": False}
            )
        return response.json()["recognized_text"]
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def create_interface():
    """Create the complete Gradio interface"""
    with gr.Blocks(theme=THEME, css=CSS, title="AI Stock Advisor Pro") as app:
        gr.Markdown("""
        # 📈 AI Stock Advisor Pro
        *Next-Generation Financial Analysis Platform*
        """)
        
        # System Status Row
        create_system_status()
        
        # Create a record button for voice input
        record_button = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input", visible=False)
        
        # Main Tabs
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("💬 Advisor Chat"):
                chat_components = create_chat_interface()
                chat_history, chat_input, submit_btn, voice_btn, audio_toggle, stock_selector, sentiment_plot, media_display, voice_output = chat_components
                
                # Event handling for text input
                submit_btn.click(
                    fn=lambda q, cid, a: websocket_chat_handler(q, cid, a),
                    inputs=[chat_input, gr.State(str(uuid.uuid4())), audio_toggle],
                    outputs=[chat_input, gr.State(), chat_history, gr.State(), voice_output]
                )
                
                # Event handling for voice button
                voice_btn.click(
                    fn=lambda: gr.update(visible=True),
                    inputs=[],
                    outputs=[record_button]
                )
                
                # Event handling for recorded audio
                record_button.change(
                    fn=handle_voice_input,
                    inputs=[record_button],
                    outputs=[chat_input]
                )
                
                # Stock selector event
                stock_selector.change(
                    fn=create_sentiment_plot,
                    inputs=[stock_selector],
                    outputs=[sentiment_plot]
                )
            
            # Analysis Tab
            with gr.Tab("📊 Deep Analysis"):
                with gr.Row():
                    analysis_stock = gr.Dropdown(SUPPORTED_STOCKS, label="Select Stock", value="NVDA")
                    analysis_btn = gr.Button("🔍 Analyze", variant="primary")
                with gr.Row():
                    analysis_sentiment_plot = gr.Plot(label="Sentiment Trend", every=300)
                    word_cloud = gr.Image(label="News Word Cloud")
                analysis_video = gr.Video(label="Market Summary Video")
                
                # Analysis stock event
                analysis_stock.change(
                    fn=create_sentiment_plot,
                    inputs=[analysis_stock],
                    outputs=[analysis_sentiment_plot]
                )
            
            # Media Tab
            with gr.Tab("🎥 Media Center"):
                gr.Markdown("### Generated Media Analysis")
                with gr.Row():
                    gr.Video(label="Market Trends")
                    gr.Image(label="Sentiment Heatmap")
                gr.Button("🔄 Refresh Media", variant="secondary")
        
        # Example queries
        gr.Markdown("### Example Queries")
        gr.Examples(
            examples=[
                "Analyze NVDA's recent market performance",
                "Compare TSLA and GOOG as long-term investments",
                "Generate video report for META",
                "What's the sentiment outlook for AAPL?"
            ],
            inputs=chat_input
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        favicon_path="favicon.ico"
    )