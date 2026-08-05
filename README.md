# Advanced RAG Agent: Arxiv Research Paper Assistant

This repository contains a production-ready Retrieval-Augmented Generation (RAG) agent, tailored for research papers and technical documents.

## Features
- **Multi-Query Retrieval**: Uses an LLM to generate multiple variants of a user query, improving vector search recall.
- **Modular Design**: Separated concerns (ingestion, retrieval, generation) in a clean `src/` structure.
- **Vector Storage**: Uses ChromaDB for local vector embeddings storage.
- **Interfaces**: Includes both a Command Line Interface (CLI) and a Streamlit Web UI.

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Add your `OPENAI_API_KEY` to the `.env` file.

### 5. Ingest Documents
Place your `.txt` or `.pdf` files in the `data/` directory. A sample paper is provided.
Run the ingestion script to populate the vector database:
```bash
python src/ingest.py
```

### 6. Run the Application
**Via CLI:**
```bash
python main_cli.py
```

**Via Streamlit Web UI:**
```bash
streamlit run app.py
```

## Testing
To run the automated tests, ensure you are in your virtual environment and run:
```bash
pytest
```
