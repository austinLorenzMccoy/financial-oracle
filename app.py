import os
import logging
import asyncio
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from contextlib import asynccontextmanager

from main import get_financial_advice_system, MessageRole, SUPPORTED_STOCKS

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and start the financial advice system
    advisor = get_financial_advice_system()
    await advisor.start()  # Start background tasks
    yield
    # Cleanup logic can be added here if needed

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Stock Trading Advisor", 
    description="RAG-based financial advice system for stock trading",
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

# Pydantic models for API
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

# Dependency to get the financial advice system
def get_advisor():
    return get_financial_advice_system()

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Stock Trading Advisor API is running. Use /docs for API documentation."}

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
        logging.error(f"Error processing question: {str(e)}")
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
        logging.error(f"Error getting system status: {str(e)}")
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

# For testing and development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)