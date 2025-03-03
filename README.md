# Enchanted Financial Oracle: A Robust RAG-based Stock Trading Advisor

Welcome to the Enchanted Financial Oracle – a state-of-the-art Retrieval-Augmented Generation (RAG) system that casts a spell of clarity over the intricate realm of stock trading. By blending the latest financial news with advanced natural language processing, our oracle offers balanced insights and thoughtful advice on major tech stocks such as NVIDIA (NVDA), Tesla (TSLA), Alphabet (GOOG), Apple (AAPL), Meta (META), and Amazon (AMZN).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Implementation Details](#implementation-details)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview

Step into a realm where robust engineering meets financial foresight. The Enchanted Financial Oracle harnesses the magic of real-time data scraping, ephemeral storage, and a dynamic RAG pipeline to deliver insights that are as enlightening as they are balanced. Whether you’re a seasoned trader or an inquisitive newcomer, our system is designed to provide context-aware advice—always with a generous sprinkle of transparency and a hearty disclaimer: **This is not professional financial advice.**

## Features

- **Automated News Alchemy**: Every 6 hours, our web scrapers conjure the latest financial news from esteemed sources like Yahoo Finance, MarketWatch, and Seeking Alpha.
- **Ephemeral Data Repository**: Fresh insights are kept in a temporary, in-memory repository that cleanses itself every 24 hours.
- **Spellbinding Vector Store**: Leveraging FAISS and HuggingFace’s all-MiniLM-L6-v2 embeddings, the system transforms raw text into a treasure trove of retrievable context.
- **Dynamic RAG Pipeline**: Merges retrieved context with a sophisticated Groq LLM to produce balanced and insightful financial guidance.
- **Versatile Interfaces**: Choose your channel—RESTful API, WebSocket real-time chat, or an interactive Gradio interface—to interact with the oracle.
- **Elegant Logging & Error Handling**: Detailed logs and robust error handling ensure that every transaction is as transparent as it is dependable.

## System Architecture

The Enchanted Financial Oracle is powered by a modular and scalable design:

1. **Web Scraper**: Gathers the latest market news from diverse sources.
2. **News Repository**: A temporary in-memory store that holds fresh financial articles and automatically purges outdated content.
3. **Vector Store**: A FAISS-based magical archive that converts text into powerful embeddings, enabling rapid and relevant retrieval.
4. **RAG Pipeline**: Combines real-time context with generative insights from a Groq LLM to craft thoughtful responses.
5. **API & WebSocket Layer**: Built on FastAPI, these interfaces ensure high-performance access and real-time communication.

## Installation

To invoke the powers of the oracle, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/financial-oracle.git
   cd financial-oracle
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment Variables**:
   Create a `.env` file in the project root and add your GROQ API key:
   ```
   GROQ_API_KEY=your-groq-api-key
   ```

## Usage

Launch the oracle with a single command:

```bash
python app.py
```

For an interactive, visually engaging experience, try the Gradio interface:

```bash
python gradio_app.py
```

## API Endpoints

Engage with the oracle through these powerful endpoints:

- **GET /**  
  Check if the API is up and running.

- **POST /api/ask**  
  Submit your stock-related questions and receive balanced advice.

- **GET /api/news/{stock_symbol}**  
  Retrieve the freshest news for a specific stock symbol.

- **GET /api/status**  
  Monitor the current status and health of the system.

- **WebSocket /ws/chat/{conversation_id}**  
  Enjoy a real-time conversation with the oracle.

### Example API Request

```python
import requests

response = requests.post("http://localhost:8000/api/ask", 
                         json={"question": "Should I buy NVDA stock given the current trends?"})
print(response.json())
```

## Implementation Details

### Data Scraping Mechanism

Our oracle employs a relentless web scraper that gathers and refines financial news from multiple reputable sources every 6 hours. This ensures that the insights you receive are always based on the very latest market developments.

### Temporary Data Storage

To maintain a dynamic and relevant dataset:
- News articles are stored temporarily in an in-memory repository.
- A background task automatically purges content older than 24 hours, ensuring that only the freshest insights are utilized.

### Vector Store for Text Retrieval

By utilizing FAISS alongside HuggingFace embeddings, the system:
- Splits news text into context-rich chunks.
- Continuously updates its vector store as new articles are scraped, ensuring rapid and relevant retrieval of information.

### Financial Advice Query System

At the heart of the oracle lies the RAG pipeline:
1. Accepts natural language queries regarding stocks.
2. Extracts context from both recent conversation history and the vector store.
3. Generates balanced, thoughtful financial advice that weighs both pros and cons.
4. Always includes a clear disclaimer to remind users that these insights are for informational purposes only.

### System Deployment

Designed with scalability and ease-of-use in mind:
- **FastAPI** powers a high-performance backend.
- **WebSockets** enable real-time interactive chat.
- **CORS Middleware** allows seamless integration with frontend applications.
- Comprehensive logging ensures every operation is meticulously recorded for transparency and debugging.

## Future Enhancements

The journey of the Enchanted Financial Oracle is ongoing. Future enhancements include:
- **Voice-Activated Interactions**: Integrating speech recognition for a hands-free experience.
- **Visual Data Narratives**: Adding video-based explanations to visually illustrate market trends.
- **Enhanced Sentiment Analysis**: Tailoring advice more personally by deciphering user sentiment.
- **Expanded Data Sources**: Integrating additional financial APIs to enrich the data pool.
- **Broader Stock Coverage**: Extending advisory services to encompass a wider range of stocks.

## Contributing

Your contributions can help shape the future of this financial oracle! Whether you’re submitting bug fixes, new features, or documentation improvements, please feel free to fork the repository and open a pull request. For major changes, open an issue first to discuss your ideas.


---

Embrace the magic of data and let the Enchanted Financial Oracle illuminate your journey through the labyrinth of stock trading. Happy investing, and always remember: even the wisest oracle advises caution!


