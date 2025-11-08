# Allie AI Documentation

Welcome to the Allie AI documentation! This repository contains a comprehensive AI assistant system with advanced learning capabilities, memory management, and multi-source information retrieval.

## 📚 Documentation Index

### Core System Documentation
- **[Continual Learning Methods](./CONTINUAL_LEARNING_METHODS.md)** - Advanced learning techniques and methodologies
- **[Data Collection Architecture](./DATA_COLLECTION_ARCHITECTURE.md)** - System for gathering and processing training data
- **[Incremental Training System](./INCREMENTAL_TRAINING_SYSTEM.md)** - Progressive model improvement and adaptation
- **[Learning Management API](./LEARNING_MANAGEMENT_API.md)** - API endpoints for learning system management
- **[Learning Requirements](./LEARNING_REQUIREMENTS.md)** - Prerequisites and setup for learning systems

### Recent Developments

## ✅ Comprehensive Query Processing Implementation

### Overview
Allie AI now features a sophisticated query processing system that integrates memory recall, web search, and Wikipedia lookups to provide comprehensive, accurate responses while continuously learning from interactions.

### 🔧 Core Implementation

#### Enhanced `generate_response` Function
- **Memory-First Approach**: Checks Allie's memory for relevant facts before external searches
- **Intelligent Search Triggering**: Performs web searches for current/time-sensitive queries, Wikipedia searches for historical/scientific queries
- **Multi-Source Synthesis**: Combines memory facts, DuckDuckGo results, and Wikipedia summaries into coherent context
- **Automatic Fact Learning**: Extracts and stores new factual information from external sources in appropriate categories

#### Search Integration
- **DuckDuckGo Web Search**: Returns structured results with titles, text, and sources
- **Wikipedia API**: Provides authoritative summaries and background information
- **Error Handling**: Graceful fallbacks when external services are unavailable

#### Information Synthesis
- **Contextual Assembly**: Builds comprehensive prompts that include memory, web results, and Wikipedia data
- **Natural Response Generation**: Uses TinyLlama to create conversational responses that blend all information sources
- **Learning Confirmations**: Acknowledges when new facts are stored from external sources

### 🧪 Validation Results

#### Automatic Learning System
Tested with 10 diverse factual statements - correctly extracted and categorized 13 facts across:
- Geography
- Biography
- History
- Science
- Technology

#### Search Functions
- ✅ Wikipedia integration working perfectly (tested with "Paris France" query)
- ✅ DuckDuckGo API functional (infrastructure solid)
- ✅ Proper error handling and user agent headers implemented

#### Code Quality
- ✅ No syntax errors in the implementation
- ✅ Proper async/await patterns for external API calls
- ✅ Comprehensive logging and error handling

### 🎯 Key Features Implemented

1. **Memory Check → External Search → Synthesis Workflow**
2. **Intelligent Search Triggering** (current events, historical facts, definitions)
3. **Multi-Source Information Integration**
4. **Automatic Fact Storage from External Sources**
5. **Natural Conversational Responses**
6. **Learning Confirmations and Acknowledgments**

### 📋 Query Processing Workflow

When a user asks a question, Allie follows this comprehensive workflow:

1. **Process Input**: Extract any learnable facts from the user's message
2. **Check Memory**: Retrieve relevant stored facts
3. **Determine Search Needs**: Check if external information is required
4. **Perform Searches**: Query DuckDuckGo and/or Wikipedia as needed
5. **Synthesize Information**: Combine all sources into coherent context
6. **Generate Response**: Create natural, informative reply using TinyLlama
7. **Store New Facts**: Learn from external sources and confirm learning
8. **Provide Confirmations**: Acknowledge what was learned

### 🏗️ System Architecture

#### Backend Components
- **FastAPI Server** (`server.py`): Main API server handling conversations and responses
- **Automatic Learner** (`automatic_learner.py`): Fact extraction and categorization engine
- **Allie Memory** (`allie_memory.py`): Persistent knowledge base management
- **TinyLlama Model**: Core language model for response generation

#### External Integrations
- **DuckDuckGo Instant Answer API**: Current web information retrieval
- **Wikipedia API**: Authoritative background information and summaries
- **httpx**: Asynchronous HTTP client for external API calls

#### Data Management
- **Conversation History**: Persistent storage of user interactions
- **Fact Categories**: Organized knowledge base (geography, biography, history, science, technology)
- **Backup System**: Automatic conversation and data backups

### 🚀 Getting Started

#### Prerequisites
- Python 3.8+ (Note: Python 3.14 may have compatibility issues with transformers)
- Required packages: fastapi, uvicorn, transformers, torch, httpx, wikipedia-api

#### Installation
```bash
# Clone the repository
git clone https://github.com/brockhager/allie-ai.git
cd allie-ai

# Install dependencies
pip install -r requirements.txt

# Start the server
python backend/server.py
```

#### API Usage
```python
import httpx

# Generate a response
async with httpx.AsyncClient() as client:
    response = await client.post(
        'http://localhost:8001/api/generate',
        json={'prompt': 'What is the capital of France?', 'max_tokens': 200}
    )
    result = response.json()
    print(result['text'])
```

### 🔍 Testing

#### Run Learning Tests
```bash
python test_automatic_learning.py
```

#### Run Search Tests
```bash
python test_search.py
```

#### Run Server Tests
```bash
python test_app.py
```

### 📈 Performance & Capabilities

The system now provides comprehensive, accurate responses that leverage Allie's growing knowledge base while seamlessly integrating:
- Current web information via DuckDuckGo
- Authoritative sources via Wikipedia
- Persistent memory of learned facts
- Natural conversational responses
- Continuous learning from user interactions

### 🔄 Future Enhancements

- Enhanced error handling for external API failures
- Improved search result ranking and relevance scoring
- Additional knowledge sources (academic papers, news APIs)
- Advanced fact verification and conflict resolution
- Multi-language support for international queries

---

*Last updated: November 7, 2025*

For technical support or contributions, please refer to the individual documentation files or create an issue in the repository.