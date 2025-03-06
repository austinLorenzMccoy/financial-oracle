import os
import logging
import asyncio
import uuid
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, Response, BackgroundTasks, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from contextlib import asynccontextmanager
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import cv2
import soundfile as sf
import sounddevice as sd

# Project imports
from main import get_financial_advice_system, MessageRole, SUPPORTED_STOCKS

# Import media components from media.py
from media import (
    SentimentAnalyzer, 
    NewsAnalyzer, 
    StockVisualizer, 
    VoiceInteraction, 
    ARTIFACTS_DIR
)

# Ensure ARTIFACTS_DIR exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Initialize NLTK resources
nltk.download('vader_lexicon', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and start the financial advice system
    advisor = get_financial_advice_system()
    await advisor.start()  # Start background tasks
    
    # Initialize media components
    app.state.visualizer = StockVisualizer()
    app.state.voice_interaction = VoiceInteraction()
    app.state.news_analyzer = NewsAnalyzer(advisor)
    
    # Log initialization
    logger.info("System started and media components initialized")
    
    yield
    
    # Cleanup logic
    plt.close('all')  # Close any remaining matplotlib figures
    logger.info("System shutdown complete")

app = FastAPI(
    title="Stock Trading Advisor with Media Analysis", 
    description="RAG-based financial advice system with advanced media analysis capabilities",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing Pydantic models
class QueryInput(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    conversation_id: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.now)

class StockNewsItem(BaseModel):
    title: str
    content: str
    source: str
    url: str
    timestamp: str

class StockNews(BaseModel):
    stock_symbol: str
    news: List[StockNewsItem]

class SystemStatus(BaseModel):
    status: str
    supported_stocks: List[str]
    last_scrape_time: Optional[datetime] = None
    total_news_items: int
    vector_store_size: int

# Enhanced media analysis models
class SentimentAnalysis(BaseModel):
    stock_symbol: str
    total_news_items: int
    average_polarity: float
    sentiment_distribution: Dict[str, int]
    sample_news: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)

class MediaOutput(BaseModel):
    stock_symbol: str
    media_type: str
    file_path: str
    base64_data: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class AudioConfig(BaseModel):
    text: str
    save_audio: bool = True
    return_base64: bool = True

class AudioResponse(BaseModel):
    text: str
    audio_path: Optional[str] = None
    base64_audio: Optional[str] = None
    duration: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class SpeechRecognitionConfig(BaseModel):
    timeout: int = 5
    save_recording: bool = True
    return_base64: bool = False

class SpeechRecognitionResponse(BaseModel):
    recognized_text: str
    audio_path: Optional[str] = None 
    base64_audio: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class MediaAnalysisRequest(BaseModel):
    stock_symbol: str
    generate_plots: bool = True
    generate_videos: bool = False
    include_news: bool = True
    include_sentiment: bool = True

class MediaAnalysisResponse(BaseModel):
    stock_symbol: str
    analysis: Dict[str, Any]
    media_files: List[Dict[str, str]]
    timestamp: datetime = Field(default_factory=datetime.now)

# Dependency setup
def get_advisor():
    return get_financial_advice_system()

def get_news_analyzer(advisor = Depends(get_advisor)):
    return NewsAnalyzer(advisor)

def get_visualizer():
    return StockVisualizer()

def get_voice_interaction():
    return VoiceInteraction()

# Helper functions
def file_to_base64(file_path: str) -> Optional[str]:
    """Convert a file to base64 encoding"""
    try:
        with open(file_path, "rb") as file:
            encoded_bytes = base64.b64encode(file.read())
            return encoded_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting file to base64: {e}")
        return None

# Existing endpoints from original app.py
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {"message": "Stock Trading Advisor API with Media Analysis is running. Use /docs for API documentation."}

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(query: QueryInput, advisor = Depends(get_advisor)):
    """
    Ask a question about stocks (NVDA, TSLA, GOOG, AAPL, META, AMZN) and receive financial advice.
    
    - **question**: Your question about stocks (e.g., "Should I buy NVDA stock?")
    - **conversation_id**: (Optional) ID to maintain conversation context
    """
    try:
        conversation_id = query.conversation_id or str(uuid.uuid4())
        response = await advisor.generate_response(query.question, conversation_id)
        
        return QueryResponse(
            conversation_id=conversation_id,
            response=response
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )

@app.get("/api/news/{stock_symbol}", response_model=StockNews)
async def get_stock_news(stock_symbol: str, advisor = Depends(get_advisor)):
    """
    Get latest news for a specific stock.
    
    - **stock_symbol**: Stock symbol (NVDA, TSLA, GOOG, AAPL, META, AMZN)
    """
    stock_symbol = stock_symbol.upper()
    if stock_symbol not in SUPPORTED_STOCKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported stock symbol. Please use one of: {', '.join(SUPPORTED_STOCKS)}"
        )
    
    news_items = advisor.get_stock_news(stock_symbol)
    
    return StockNews(
        stock_symbol=stock_symbol,
        news=news_items
    )

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status(advisor = Depends(get_advisor)):
    """Get the current status of the financial advice system."""
    try:
        # Get the latest scrape time
        if advisor.news_repository.news_data:
            last_scrape_time = max(item.timestamp for item in advisor.news_repository.news_data)
        else:
            last_scrape_time = None
        
        # Get vector store size
        vector_store_size = len(advisor.vectorstore.index_to_docstore_id) if hasattr(advisor, 'vectorstore') else 0
        
        return SystemStatus(
            status="running",
            supported_stocks=list(SUPPORTED_STOCKS),
            last_scrape_time=last_scrape_time,
            total_news_items=len(advisor.news_repository.news_data),
            vector_store_size=vector_store_size
        )
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )

@app.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str, advisor = Depends(get_advisor)):
    """WebSocket endpoint for real-time chat with the financial advisor."""
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": f"Welcome to the Stock Trading Advisor! Ask me about {', '.join(SUPPORTED_STOCKS)} stocks."
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Check for exit messages
            if data.lower() in {"exit", "quit", "bye", "goodbye"}:
                await websocket.send_json({
                    "type": "message",
                    "message": "Thank you for using Stock Trading Advisor. Goodbye!"
                })
                break
            
            # Log received message
            logging.info(f"Received message from {conversation_id}: {data}")
            
            # Generate response
            try:
                response = await advisor.generate_response(data, conversation_id)
                await websocket.send_json({
                    "type": "message",
                    "message": response
                })
            except Exception as e:
                logging.error(f"Error generating response: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "I'm sorry, an error occurred while processing your request."
                })
                
    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logging.error(f"Error in WebSocket: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "An unexpected error occurred. Please try again later."
            })
        except:
            # If we can't send the error message, the connection is probably already closed
            pass

# New media analysis endpoints
@app.get("/api/sentiment/{stock_symbol}", response_model=SentimentAnalysis)
async def analyze_stock_sentiment(
    stock_symbol: str, 
    news_analyzer = Depends(get_news_analyzer)
):
    """
    Get comprehensive sentiment analysis for a stock's news
    
    - **stock_symbol**: Stock symbol (NVDA, TSLA, GOOG, AAPL, META, AMZN)
    """
    try:
        stock_symbol = stock_symbol.upper()
        if stock_symbol not in SUPPORTED_STOCKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Unsupported stock symbol. Please use one of: {', '.join(SUPPORTED_STOCKS)}"
            )

        # Get full news analysis
        analysis = await news_analyzer.analyze_stock_news(stock_symbol)
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=analysis['error']
            )

        # Format response
        return SentimentAnalysis(
            stock_symbol=stock_symbol,
            total_news_items=analysis['total_news_items'],
            average_polarity=analysis['sentiment_summary']['avg_polarity'],
            sentiment_distribution=analysis['sentiment_summary']['sentiment_distribution'],
            sample_news=analysis['news_samples']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform sentiment analysis: {str(e)}"
        )

@app.get("/api/sentiment/plot/{stock_symbol}")
async def get_sentiment_plot(
    stock_symbol: str,
    return_base64: bool = Query(False, description="Return image as base64 encoded string"),
    news_analyzer = Depends(get_news_analyzer),
    visualizer = Depends(get_visualizer)
):
    """
    Get visualization of news sentiment distribution as PNG image
    
    - **stock_symbol**: Stock symbol (NVDA, TSLA, GOOG, AAPL, META, AMZN)
    - **return_base64**: Set to true to receive the image as base64 encoded string
    """
    try:
        stock_symbol = stock_symbol.upper()
        if stock_symbol not in SUPPORTED_STOCKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Unsupported stock symbol. Please use one of: {', '.join(SUPPORTED_STOCKS)}"
            )
            
        # Get analysis
        analysis = await news_analyzer.analyze_stock_news(stock_symbol)
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=analysis['error']
            )
            
        # Generate plot
        image_path = visualizer.plot_sentiment(analysis)
        
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate visualization"
            )
            
        # Return as base64 if requested
        if return_base64:
            base64_data = file_to_base64(image_path)
            return {
                "stock_symbol": stock_symbol,
                "file_path": image_path,
                "base64_image": base64_data
            }
            
        # Otherwise return the image directly
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        return Response(
            content=image_bytes, 
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={stock_symbol}_sentiment.png"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plot generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate visualization: {str(e)}"
        )

@app.get("/api/sentiment/video/{stock_symbol}")
async def get_sentiment_video(
    stock_symbol: str,
    video_type: str = Query("trend", description="Type of video: 'trend' or 'news'"),
    return_base64: bool = Query(False, description="Return video as base64 encoded string"),
    news_analyzer = Depends(get_news_analyzer),
    visualizer = Depends(get_visualizer)
):
    """
    Get video visualization of news sentiment for a stock
    
    - **stock_symbol**: Stock symbol (NVDA, TSLA, GOOG, AAPL, META, AMZN)
    - **video_type**: Type of video: 'trend' for sentiment trend or 'news' for news summary
    - **return_base64**: Set to true to receive the video as base64 encoded string
    """
    try:
        stock_symbol = stock_symbol.upper()
        if stock_symbol not in SUPPORTED_STOCKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Unsupported stock symbol. Please use one of: {', '.join(SUPPORTED_STOCKS)}"
            )
            
        # Get analysis
        analysis = await news_analyzer.analyze_stock_news(stock_symbol)
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=analysis['error']
            )
            
        # Generate video based on type
        video_path = ""
        if video_type.lower() == "trend":
            video_path = visualizer.create_sentiment_trend_video(analysis)
        elif video_type.lower() == "news":
            video_path = visualizer.create_news_summary_video(analysis)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid video_type. Use 'trend' or 'news'"
            )
            
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate video"
            )
            
        # Return as base64 if requested
        if return_base64:
            base64_data = file_to_base64(video_path)
            return {
                "stock_symbol": stock_symbol,
                "video_type": video_type,
                "file_path": video_path,
                "base64_video": base64_data
            }
            
        # Otherwise return the video directly
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            
        return Response(
            content=video_bytes, 
            media_type="video/mp4",
            headers={"Content-Disposition": f"inline; filename={stock_symbol}_{video_type}.mp4"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate video: {str(e)}"
        )

@app.post("/api/tts", response_model=AudioResponse)
async def text_to_speech(
    config: AudioConfig,
    background_tasks: BackgroundTasks,
    voice_interaction = Depends(get_voice_interaction)
):
    """
    Convert text to speech
    
    - **text**: Text to convert to speech
    - **save_audio**: Whether to save the audio to disk
    - **return_base64**: Whether to return the audio as base64 encoded string
    """
    try:
        # Generate a timestamp-based ID for the audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(voice_interaction.recordings_dir, f"api_response_{timestamp}.wav")
        
        # Generate speech
        voice_interaction.tts_engine.save_to_file(config.text, audio_path)
        voice_interaction.tts_engine.runAndWait()
        
        # Get audio duration
        with wave.open(audio_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
        
        # Prepare the response
        response = AudioResponse(
            text=config.text,
            audio_path=audio_path if config.save_audio else None,
            duration=duration
        )
        
        # Add base64 data if requested
        if config.return_base64:
            response.base64_audio = file_to_base64(audio_path)
        
        # Clean up the file if not saving
        if not config.save_audio:
            background_tasks.add_task(os.remove, audio_path)
        
        return response
    except Exception as e:
        logger.error(f"Text-to-speech failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert text to speech: {str(e)}"
        )

@app.post("/api/speech-recognition", response_model=SpeechRecognitionResponse)
async def speech_recognition(
    file: UploadFile = File(...),
    save_recording: bool = Query(True, description="Save the audio recording"),
    return_base64: bool = Query(False, description="Return audio as base64 string"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    voice_interaction = Depends(get_voice_interaction)
):
    """
    Perform speech recognition on an uploaded audio file
    
    - **file**: Audio file to transcribe (WAV format recommended)
    - **save_recording**: Whether to save the recording to disk
    - **return_base64**: Whether to return the audio as base64 encoded string
    """
    try:
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(voice_interaction.recordings_dir, f"api_upload_{timestamp}.wav")
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Perform speech recognition
        recognizer = voice_interaction.recognizer
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        
        # Prepare response
        response = SpeechRecognitionResponse(
            recognized_text=text,
            audio_path=file_path if save_recording else None
        )
        
        # Add base64 data if requested
        if return_base64:
            response.base64_audio = file_to_base64(file_path)
        
        # Clean up if not saving
        if not save_recording:
            background_tasks.add_task(os.remove, file_path)
        
        return response
    except sr.UnknownValueError:
        return SpeechRecognitionResponse(recognized_text="")
    except Exception as e:
        logger.error(f"Speech recognition failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform speech recognition: {str(e)}"
        )

@app.get("/api/audio-devices")
async def list_audio_devices(voice_interaction = Depends(get_voice_interaction)):
    """
    List available audio input and output devices
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        formatted_devices = []
        
        for i, device in enumerate(devices):
            device_info = {
                "id": i,
                "name": device["name"],
                "hostapi": device["hostapi"],
                "max_input_channels": device["max_input_channels"],
                "max_output_channels": device["max_output_channels"],
                "default_samplerate": device["default_samplerate"],
                "type": "Input" if device["max_input_channels"] > 0 else "Output"
            }
            formatted_devices.append(device_info)
            
        return {
            "devices": formatted_devices,
            "default_input": sd.default.device[0],
            "default_output": sd.default.device[1]
        }
    except Exception as e:
        logger.error(f"Error listing audio devices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list audio devices: {str(e)}"
        )

@app.post("/api/set-audio-devices")
async def set_audio_devices(
    input_device: Optional[int] = Query(None, description="Input device ID"),
    output_device: Optional[int] = Query(None, description="Output device ID"),
    voice_interaction = Depends(get_voice_interaction)
):
    """
    Set the audio input and output devices
    
    - **input_device**: ID of the input device (microphone)
    - **output_device**: ID of the output device (speaker)
    """
    try:
        voice_interaction.set_devices(mic_index=input_device, speaker_index=output_device)
        return {"message": "Audio devices configured successfully"}
    except Exception as e:
        logger.error(f"Error setting audio devices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set audio devices: {str(e)}"
        )

@app.post("/api/media-analysis", response_model=MediaAnalysisResponse)
async def full_media_analysis(
    request: MediaAnalysisRequest,
    news_analyzer = Depends(get_news_analyzer),
    visualizer = Depends(get_visualizer)
):
    """
    Perform a comprehensive media analysis for a stock, including sentiment analysis,
    visualizations, and videos
    
    - **stock_symbol**: Stock symbol (NVDA, TSLA, GOOG, AAPL, META, AMZN)
    - **generate_plots**: Whether to generate static plots
    - **generate_videos**: Whether to generate videos
    - **include_news**: Whether to include news items in the response
    - **include_sentiment**: Whether to include sentiment analysis in the response
    """
    try:
        stock_symbol = request.stock_symbol.upper()
        if stock_symbol not in SUPPORTED_STOCKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Unsupported stock symbol. Please use one of: {', '.join(SUPPORTED_STOCKS)}"
            )
            
        # Get full analysis
        analysis = await news_analyzer.analyze_stock_news(stock_symbol)
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=analysis['error']
            )
            
        # Create simplified response
        response_data = {
            "stock_symbol": stock_symbol,
            "analysis": {},
            "media_files": []
        }
        
        # Include sentiment analysis if requested
        if request.include_sentiment:
            response_data["analysis"]["sentiment"] = {
                "average_polarity": analysis["sentiment_summary"]["avg_polarity"],
                "distribution": analysis["sentiment_summary"]["sentiment_distribution"]
            }
            
        # Include news if requested
        if request.include_news:
            response_data["analysis"]["news"] = analysis["news_samples"]
            
        # Generate plots if requested
        if request.generate_plots:
            plot_path = visualizer.plot_sentiment(analysis)
            if plot_path:
                response_data["media_files"].append({
                    "type": "sentiment_plot",
                    "path": plot_path,
                    "mime_type": "image/png"
                })
                
        # Generate videos if requested
        if request.generate_videos:
            # Trend video
            trend_video_path = visualizer.create_sentiment_trend_video(analysis)
            if trend_video_path:
                response_data["media_files"].append({
                    "type": "trend_video",
                    "path": trend_video_path,
                    "mime_type": "video/mp4"
                })
                
            # News video
            news_video_path = visualizer.create_news_summary_video(analysis)
            if news_video_path:
                response_data["media_files"].append({
                    "type": "news_video",
                    "path": news_video_path,
                    "mime_type": "video/mp4"
                })
                
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full media analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform media analysis: {str(e)}"
        )

@app.websocket("/ws/voice-chat/{conversation_id}")
async def websocket_voice_chat(
    websocket: WebSocket, 
    conversation_id: str,
    advisor = Depends(get_advisor),
    voice_interaction = Depends(get_voice_interaction)
):
    """WebSocket endpoint for voice-based chat with the financial advisor."""
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": f"Welcome to the Voice-Enabled Stock Trading Advisor! You can communicate via voice or text."
        })
        
        while True:
            # Receive message (text or voice command)
            data = await websocket.receive_json()
            
            # Check for command type
            command_type = data.get("type", "text")
            
            if command_type == "exit":
                await websocket.send_json({
                    "type": "message",
                    "message": "Thank you for using Voice-Enabled Stock Trading Advisor. Goodbye!"
                })
                break
                
            # Process different command types
            if command_type == "text":
                # Handle text message
                text_input = data.get("message", "")
                if not text_input:
                    await websocket.send_json({"type": "error", "message": "Empty message received"})
                    continue
                    
                # Generate response
                response = await advisor.generate_response(text_input, conversation_id)
                
                # Send text response
                await websocket.send_json({
                    "type": "text_response",
                    "message": response
                })
                
                # Generate audio if requested
                if data.get("generate_audio", False):
                    # Create unique audio file name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_path = os.path.join(voice_interaction.recordings_dir, f"ws_response_{timestamp}.wav")
                    
                    # Generate speech
                    voice_interaction.tts_engine.save_to_file(response, audio_path)
                    voice_interaction.tts_engine.runAndWait()
                    
                    # Send audio as base64
                    base64_audio = file_to_base64(audio_path)
                    await websocket.send_json({
                        "type": "audio_response",
                        "base64_audio": base64_audio
                    })
                    
                    # Clean up file
                    if not data.get("save_audio", False):
                        os.remove(audio_path)
                        
            elif command_type == "voice":
                # Handle voice data
                voice_data = data.get("base64_audio", "")
                if not voice_data:
                    await websocket.send_json({"type": "error", "message": "No audio data received"})
                    continue
                    
                try:
                    # Save audio to a temporary file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_input_path = os.path.join(voice_interaction.recordings_dir, f"ws_input_{timestamp}.wav")
                    
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(voice_data)
                    with open(temp_input_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Perform speech recognition
                    with sr.AudioFile(temp_input_path) as source:
                        audio = voice_interaction.recognizer.record(source)
                        text = voice_interaction.recognizer.recognize_google(audio)
                        
                    # Generate response
                    response = await advisor.generate_response(text, conversation_id)
                    
                    # Send text response
                    await websocket.send_json({
                        "type": "text_response",
                        "message": response,
                        "original_text": text
                    })
                    
                    # Generate response audio
                    temp_output_path = os.path.join(voice_interaction.recordings_dir, f"ws_output_{timestamp}.wav")
                    voice_interaction.tts_engine.save_to_file(response, temp_output_path)
                    voice_interaction.tts_engine.runAndWait()
                    
                    # Send audio response
                    with open(temp_output_path, "rb") as f:
                        response_audio = base64.b64encode(f.read()).decode("utf-8")
                    
                    await websocket.send_json({
                        "type": "audio_response",
                        "base64_audio": response_audio
                    })
                    
                except sr.UnknownValueError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Could not understand audio"
                    })
                except sr.RequestError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Speech recognition error: {str(e)}"
                    })
                except Exception as e:
                    logger.error(f"Voice processing error: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Error processing voice request"
                    })
                finally:
                    # Cleanup temporary files
                    for path in [temp_input_path, temp_output_path]:
                        if path and os.path.exists(path):
                            os.remove(path)
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid command type: {command_type}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket voice chat disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket voice chat: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "An unexpected error occurred. Please try again later."
            })
        except:
            pass

# Existing main block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)