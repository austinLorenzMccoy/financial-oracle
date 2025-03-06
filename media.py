import os
import sys
import asyncio
import logging
import nltk
import base64
from datetime import datetime
import textwrap

# Data Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Natural Language Processing
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Speech Recognition and Text-to-Speech
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import soundfile as sf
import librosa
import wave

# Video Processing
import cv2
from PIL import Image, ImageDraw, ImageFont

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

# Create artifacts directory if it doesn't exist
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


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
            
            sentiments = [SentimentAnalyzer.analyze(item['title'] + " " + item['content']) for item in news_items]
            composite_scores = [s['composite_score'] for s in sentiments]
            
            # Calculate sentiment trend over time if timestamps are available
            sentiment_trend = []
            if news_items and 'timestamp' in news_items[0]:
                # Sort by timestamp
                paired_items = [(item, sent) for item, sent in zip(news_items, sentiments)]
                paired_items.sort(key=lambda x: x[0]['timestamp'])
                
                # Extract trend
                sentiment_trend = [(item['timestamp'], sent['composite_score']) 
                                  for item, sent in paired_items]
            
            return {
                'stock': stock_symbol,
                'total_news_items': len(news_items),
                'sentiment_summary': {
                    'avg_polarity': np.mean(composite_scores) if composite_scores else 0,
                    'sentiment_distribution': self._sentiment_distribution(sentiments),
                    'sentiment_trend': sentiment_trend
                },
                'news_samples': [
                    {
                        'title': item['title'], 
                        'content': item['content'][:100] + "..." if len(item['content']) > 100 else item['content'],
                        'sentiment': sent['composite_score'],
                        'source': item.get('source', 'Unknown'),
                        'timestamp': item.get('timestamp', 'Unknown')
                    }
                    for item, sent in zip(news_items[:5], sentiments[:5])
                ],
                'raw_news': news_items  # Include raw news for further processing
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
    """Enhanced visualization utilities for stock analysis with video capabilities."""
    
    def __init__(self):
        self.font_path = None
        # Try to find a font that works on most systems
        for font_path in [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:\\Windows\\Fonts\\arial.ttf'  # Windows
        ]:
            if os.path.exists(font_path):
                self.font_path = font_path
                break
    
    def plot_sentiment(self, analysis: dict, filename: str = None) -> str:
        """Generate static sentiment analysis visualization."""
        try:
            if filename is None:
                filename = os.path.join(ARTIFACTS_DIR, f"{analysis['stock']}_sentiment_{self._get_timestamp()}.png")
            
            plt.figure(figsize=(10, 6))
            dist = analysis['sentiment_summary']['sentiment_distribution']
            categories = list(dist.keys())
            values = list(dist.values())
            
            # Custom colors and improved visualization
            colors = ['darkgreen', 'lightgreen', 'gray', 'salmon', 'darkred']
            plt.bar(categories, values, color=colors)
            
            plt.title(f"{analysis['stock']} News Sentiment Analysis", fontsize=16)
            plt.xlabel("Sentiment Category", fontsize=12)
            plt.ylabel("Number of Articles", fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.1, str(v), ha='center')
                
            # Add summary statistics
            avg_polarity = analysis['sentiment_summary']['avg_polarity']
            sentiment_label = "Positive" if avg_polarity > 0.1 else "Negative" if avg_polarity < -0.1 else "Neutral"
            plt.annotate(f"Overall Sentiment: {sentiment_label} ({avg_polarity:.2f})", 
                         xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()
            
            logger.info(f"Saved sentiment plot to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return ''
    
    def create_sentiment_trend_video(self, analysis: dict, filename: str = None) -> str:
        """Generate video showing sentiment trend over time."""
        try:
            if filename is None:
                filename = os.path.join(ARTIFACTS_DIR, f"{analysis['stock']}_trend_{self._get_timestamp()}.mp4")
            
            sentiment_trend = analysis['sentiment_summary'].get('sentiment_trend', [])
            if not sentiment_trend:
                logger.warning(f"No sentiment trend data available for {analysis['stock']}")
                return ''
            
            # Sort by timestamp if needed
            sentiment_trend.sort(key=lambda x: x[0])
            timestamps = [item[0] for item in sentiment_trend]
            sentiments = [item[1] for item in sentiment_trend]
            
            # Create date labels (simplified for display)
            date_labels = []
            for ts in timestamps:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        date_labels.append(dt.strftime('%m-%d %H:%M'))
                    else:
                        date_labels.append('Unknown')
                except:
                    date_labels.append('Unknown')
            
            # Create a figure for animation
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Function to update the animation
            def update(num):
                ax.clear()
                ax.plot(date_labels[:num+1], sentiments[:num+1], 'b-o')
                ax.set_ylim(-1.1, 1.1)
                ax.set_ylabel('Sentiment Score')
                ax.set_xlabel('Date')
                ax.set_title(f"{analysis['stock']} Sentiment Trend Over Time")
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            # Create animation
            ani = animation.FuncAnimation(fig, update, frames=len(date_labels), repeat=False)
            
            # Save as mp4
            writer = animation.FFMpegWriter(fps=2, bitrate=1800)
            ani.save(filename, writer=writer)
            plt.close()
            
            logger.info(f"Saved sentiment trend video to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Video creation error: {e}", exc_info=True)
            return ''
    
    def create_news_summary_video(self, analysis: dict, filename: str = None) -> str:
        """Generate video showing key news with sentiment scores."""
        try:
            if filename is None:
                filename = os.path.join(ARTIFACTS_DIR, f"{analysis['stock']}_news_{self._get_timestamp()}.mp4")
            
            news_samples = analysis.get('news_samples', [])
            if not news_samples:
                logger.warning(f"No news samples available for {analysis['stock']}")
                return ''
            
            # Create temporary directory for frames
            temp_dir = os.path.join(ARTIFACTS_DIR, 'temp_frames')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Settings
            width, height = 800, 600
            bg_color = (255, 255, 255)  # White background
            text_color = (0, 0, 0)      # Black text
            
            frames = []
            
            # Title frame
            title_img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(title_img)
            
            # Add title
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, 36)
                subtitle_font = ImageFont.truetype(self.font_path, 24)
                text_font = ImageFont.truetype(self.font_path, 20)
            else:
                # Use default font if custom font not available
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            draw.text((width//2, 100), f"{analysis['stock']} News Analysis", 
                     fill=text_color, font=title_font, anchor="mm")
            draw.text((width//2, 160), f"Overall Sentiment: {analysis['sentiment_summary']['avg_polarity']:.2f}", 
                     fill=text_color, font=subtitle_font, anchor="mm")
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw.text((width//2, height-50), f"Generated: {timestamp}", 
                     fill=text_color, font=text_font, anchor="mm")
            
            title_frame_path = os.path.join(temp_dir, "title.png")
            title_img.save(title_frame_path)
            frames.append(title_frame_path)
            
            # News frames
            for i, news in enumerate(news_samples):
                img = Image.new('RGB', (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                
                # Determine sentiment color
                sentiment = news['sentiment']
                if sentiment > 0.25:
                    sentiment_color = (0, 150, 0)  # Green
                elif sentiment < -0.25:
                    sentiment_color = (150, 0, 0)  # Red
                else:
                    sentiment_color = (100, 100, 100)  # Gray
                
                # Draw header
                draw.text((width//2, 50), f"News #{i+1}: {news['source']}", 
                         fill=text_color, font=subtitle_font, anchor="mm")
                
                # Draw sentiment indicator
                draw.rectangle([width//2-100, 80, width//2+100, 100], outline=(0,0,0))
                sentiment_width = int(200 * (sentiment + 1) / 2)  # Scale from -1,1 to 0,200
                draw.rectangle([width//2-100, 80, width//2-100+sentiment_width, 100], 
                              fill=sentiment_color)
                draw.text((width//2, 110), f"Sentiment: {sentiment:.2f}", 
                         fill=text_color, font=text_font, anchor="mm")
                
                # Draw news title with word wrapping
                title_wrapped = textwrap.fill(news['title'], width=40)
                y_pos = 160
                for line in title_wrapped.split('\n'):
                    draw.text((width//2, y_pos), line, fill=text_color, font=text_font, anchor="mm")
                    y_pos += 30
                
                # Draw content preview with word wrapping
                if 'content' in news:
                    content_wrapped = textwrap.fill(news['content'], width=50)
                    y_pos += 20
                    draw.text((width//2, y_pos), "Summary:", fill=text_color, font=text_font, anchor="mm")
                    y_pos += 30
                    
                    for line in content_wrapped.split('\n')[:5]:  # Limit to 5 lines
                        draw.text((width//2, y_pos), line, fill=text_color, font=text_font, anchor="mm")
                        y_pos += 30
                
                # Add timestamp if available
                if 'timestamp' in news and news['timestamp'] != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(news['timestamp'].replace('Z', '+00:00'))
                        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        draw.text((width//2, height-50), f"Published: {timestamp_str}", 
                                 fill=text_color, font=text_font, anchor="mm")
                    except:
                        pass
                
                frame_path = os.path.join(temp_dir, f"frame_{i}.png")
                img.save(frame_path)
                frames.append(frame_path)
            
            # Create video from frames
            self._create_video_from_frames(frames, filename, fps=1)
            
            # Clean up temporary files
            for frame in frames:
                if os.path.exists(frame):
                    os.remove(frame)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            
            logger.info(f"Saved news summary video to {filename}")
            return filename
        except Exception as e:
            logger.error(f"News video creation error: {e}", exc_info=True)
            return ''
    
    def _create_video_from_frames(self, frame_paths, output_path, fps=1):
        """Create a video from a list of image files."""
        if not frame_paths:
            logger.error("No frames provided for video creation")
            return False
        
        try:
            frame = cv2.imread(frame_paths[0])
            height, width, layers = frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                # Hold each frame for longer (repeat frames)
                for _ in range(3):  # Each frame appears for 3 seconds at 1 fps
                    video.write(frame)
            
            cv2.destroyAllWindows()
            video.release()
            return True
        except Exception as e:
            logger.error(f"Error creating video: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _get_timestamp():
        """Generate a timestamp string for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


class VoiceInteraction:
    """Class for handling two-way voice interaction."""
    
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        
        # Configure devices
        self.mic_index = None  # Default microphone
        self.speaker_index = None  # Default speaker
        
        # Setup recordings directory
        self.recordings_dir = os.path.join(ARTIFACTS_DIR, "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)
    
    def list_audio_devices(self):
        """List available input and output audio devices."""
        try:
            devices = sd.query_devices()
            print("\nAvailable Audio Devices:")
            print("-" * 50)
            for i, device in enumerate(devices):
                device_type = "Input" if device['max_input_channels'] > 0 else "Output"
                print(f"{i}: {device['name']} ({device_type})")
            print("-" * 50)
            return devices
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return []
    
    def set_devices(self, mic_index=None, speaker_index=None):
        """Set the microphone and speaker devices to use."""
        self.mic_index = mic_index
        self.speaker_index = speaker_index
        if speaker_index is not None:
            try:
                # Configure TTS to use the selected output device
                self.tts_engine.setProperty('voice', speaker_index)
                logger.info(f"Set speaker device to index {speaker_index}")
            except Exception as e:
                logger.error(f"Error setting speaker device: {e}")
    
    def listen(self, timeout=5, save_recording=True) -> str:
        """
        Listen for speech input and convert to text.
        Returns the recognized text or empty string if nothing recognized.
        """
        try:
            print("\nListening... (Speak now)")
            
            # Record audio
            recording_path = None
            if save_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recording_path = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
            
            with sr.Microphone(device_index=self.mic_index) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                try:
                    audio = self.recognizer.listen(source, timeout=timeout)
                    
                    # Save recording if requested
                    if save_recording and recording_path:
                        with open(recording_path, "wb") as f:
                            f.write(audio.get_wav_data())
                        logger.info(f"Saved recording to {recording_path}")
                    
                    # Convert speech to text
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    return text
                except sr.WaitTimeoutError:
                    print("No speech detected within timeout period.")
                    return ""
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return ""
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    return ""
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return ""
    
    def speak(self, text, save_audio=True):
        """
        Convert text to speech and play it.
        Optionally save the audio to a file.
        """
        try:
            print(f"\nSpeaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Save to file if requested
            if save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = os.path.join(self.recordings_dir, f"response_{timestamp}.wav")
                
                self.tts_engine.save_to_file(text, audio_path)
                self.tts_engine.runAndWait()
                logger.info(f"Saved TTS audio to {audio_path}")
                
                # Now play the file
                data, fs = sf.read(audio_path)
                sd.play(data, fs)
                sd.wait()
            else:
                # Just play directly
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
            return True
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}", exc_info=True)
            return False


class StockMediaApp:
    """Main application integrating all media components."""
    
    def __init__(self):
        # Initialize the financial advice system
        self.system = get_financial_advice_system()
        
        # Initialize components
        self.news_analyzer = NewsAnalyzer(self.system)
        self.visualizer = StockVisualizer()
        self.voice_interaction = VoiceInteraction()
        
        # Conversation tracking
        self.conversation_id = None
        
    async def start(self):
        """Start the system and initialize components."""
        try:
            # Start the financial advice system
            await self.system.start()
            
            # List available audio devices
            self.voice_interaction.list_audio_devices()
            
            # Create conversation ID
            self.conversation_id = str(datetime.now().strftime("%Y%m%d%H%M%S"))
            
            print("\n" + "=" * 50)
            print(" Stock Media Advisor with Voice Interaction")
            print("=" * 50)
            print("System initialized and ready!")
            print("You can now ask questions about stocks via voice or text.")
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            raise
    
    async def voice_interaction_loop(self):
        """Run a loop for voice-based interaction."""
        print("\nStarting voice interaction mode. Say 'exit' or 'quit' to end.")
        
        exit_commands = ["exit", "quit", "stop", "end"]
        
        while True:
            try:
                # Listen for a command
                command = self.voice_interaction.listen()
                
                # Check for exit command
                if any(exit_word in command.lower() for exit_word in exit_commands):
                    self.voice_interaction.speak("Goodbye!")
                    break
                
                # Skip if empty command
                if not command:
                    continue
                
                # Process the query
                response = await self.system.generate_response(command, self.conversation_id)
                
                # Read response aloud
                self.voice_interaction.speak(response)
                
                # Check if stock-specific query 
                mentioned_stocks = self.system.extract_stock_symbols(command)
                
                # Generate visuals if specific stocks mentioned
                if mentioned_stocks:
                    for stock in mentioned_stocks:
                        print(f"\nGenerating visuals for {stock}...")
                        
                        # Analyze news and sentiment
                        analysis = await self.news_analyzer.analyze_stock_news(stock)
                        
                        if 'error' in analysis:
                            print(f"Error analyzing {stock}: {analysis['error']}")
                            continue
                        
                        # Generate static visualization
                        self.visualizer.plot_sentiment(analysis)
                        
                        # Generate videos
                        sentiment_video = self.visualizer.create_sentiment_trend_video(analysis)
                        news_video = self.visualizer.create_news_summary_video(analysis)
                        
                        if sentiment_video or news_video:
                            video_msg = f"I've created visual analyses for {stock} in the artifacts folder."
                            print(video_msg)
                            self.voice_interaction.speak(video_msg)
            
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in voice interaction loop: {e}")
                error_msg = "I encountered an error processing your request."
                print(error_msg)
                self.voice_interaction.speak(error_msg)
    
    async def process_stock_query(self, stock_symbol):
        """Process a query about a specific stock with enhanced media features."""
        try:
            # Analyze news and sentiment
            analysis = await self.news_analyzer.analyze_stock_news(stock_symbol)
            
            if 'error' in analysis:
                return {'error': analysis['error']}
            
            # Generate media outputs
            outputs = {
                'sentiment_plot': self.visualizer.plot_sentiment(analysis),
                'sentiment_video': self.visualizer.create_sentiment_trend_video(analysis),
                'news_video': self.visualizer.create_news_summary_video(analysis),
                'analysis': analysis
            }
            
            return outputs
        except Exception as e:
            logger.error(f"Error processing stock query: {e}")
            return {'error': str(e)}


async def main():
    """Main entry point with improved interaction options."""
    try:
        # Initialize the application
        app = StockMediaApp()
        
        # Start the system
        print("Initializing system...")
        await app.start()
        
        # Allow time for initial data collection (reduced time for demo)
        print("Performing initial data collection (10 seconds)...")
        await asyncio.sleep(10)
        
        # Choose interaction mode
        while True:
            print("\nChoose interaction mode:")
            print("1. Voice interaction")
            print("2. Text-based query")
            print("3. Generate media for specific stock")
            print("4. Quit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                # Voice interaction loop
                await app.voice_interaction_loop()
            
            elif choice == '2':
                # Text-based query
                query = input("\nEnter your question about stocks: ")
                if not query:
                    continue
                    
                response = await app.system.generate_response(query, app.conversation_id)
                print(f"\nResponse: {response}")
                
                # Ask if user wants to hear the response
                hear_response = input("Would you like to hear this response? (y/n): ").lower()
                if hear_response.startswith('y'):
                    app.voice_interaction.speak(response)
            
            elif choice == '3':
                # Generate media for specific stock
                stock = input("\nEnter stock symbol (NVDA, TSLA, GOOG, AAPL, META, AMZN): ").strip().upper()
                if stock not in {"NVDA", "TSLA", "GOOG", "AAPL", "META", "AMZN"}:
                    print(f"Unsupported stock symbol: {stock}")
                    continue
                
                print(f"Generating media for {stock}...")
                result = await app.process_stock_query(stock)
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print("\nGenerated files:")
                    for key, path in result.items():
                        if key != 'analysis' and path:
                            print(f"- {key}: {path}")
                    
                    # Summary of sentiment
                    avg_sentiment = result['analysis']['sentiment_summary']['avg_polarity']
                    sentiment_msg = f"\n{stock} has an overall sentiment of {avg_sentiment:.2f} "
                    sentiment_msg += "(positive)" if avg_sentiment > 0 else "(negative)" if avg_sentiment < 0 else "(neutral)"
                    print(sentiment_msg)
                    
                    # Ask if user wants sentiment summary spoken
                    speak_summary = input("Would you like to hear a spoken summary? (y/n): ").lower()
                    if speak_summary.startswith('y'):
                        news_count = result['analysis']['total_news_items']
                        summary = f"Analysis for {stock} based on {news_count} news items. {sentiment_msg} "
                        summary += f"The top headline is: {result['analysis']['news_samples'][0]['title']}"
                        app.voice_interaction.speak(summary)
            
            elif choice == '4':
                print("Exiting program.")
                break
            
            else:
                print("Invalid choice. Please try again.")
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Critical failure: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted")
    except Exception as e:
        logger.error(f"Critical failure: {e}")