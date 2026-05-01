# AskMyPDF - Intelligent Document Assistant

AskMyPDF is an intelligent, conversational web application that allows users to upload PDF documents and interact with them through a Q&A interface. Built with Streamlit, LangChain, and powered by Groq's fast LLM inference, it enables seamless extraction and analysis of information from large PDF files.

## Features

- **PDF Upload & Processing:** Easily upload any PDF document for automated text extraction.
- **Intelligent Chunking:** Automatically splits document text into manageable chunks for effective context retrieval.
- **Vector Embeddings & Search:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and ChromaDB for fast, accurate semantic search.
- **Powered by LLaMA 3:** Utilizes the robust `llama-3.3-70b-versatile` model via Groq for high-quality, precise answers.
- **Conversational Interface:** Features a modern chat UI with session history, allowing for fluid and continuous follow-up questions.
- **Real-Time Feedback:** Includes interactive processing statuses and spinners to provide transparency during document analysis.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM Framework:** [LangChain](https://langchain.com/) (including `langchain-community`, `langchain-classic`, and `langchain-text-splitters`)
- **LLM Provider:** [Groq](https://groq.com/) API
- **Embeddings:** HuggingFace (`sentence-transformers`)
- **Vector Database:** ChromaDB
- **PDF Parsing:** PyPDF

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd smart-doc-scanner
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv311
   # On Windows:
   .venv311\Scripts\activate
   # On Linux/macOS:
   source .venv311/bin/activate
   ```

3. **Install dependencies:**
   Ensure you have the required packages installed. You can install them via pip:
   ```bash
   pip install streamlit langchain langchain-community langchain-classic langchain-groq langchain-text-splitters chromadb sentence-transformers pypdf python-dotenv
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

Start the Streamlit application by running:
```bash
streamlit run app.py
```

1. Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Upload a PDF document using the file uploader.
3. Wait for the processing to complete (the app will extract text, chunk it, and build the vector database).
4. Start asking questions about the uploaded PDF in the chat interface!

## Project Structure

- `app.py`: The main Streamlit application script containing the UI and LLM integration logic.
- `.env`: Environment variables file (ignored in version control).
- `temp.pdf`: Temporary storage for the uploaded PDF being processed.
- `files/`: Additional project files directory.
