# MemoMind AI

**Your personal AI memory assistant.**

MemoMind AI captures spoken memories through Omi, stores them in Qdrant, and uses semantic search and Lyzr AI to answer questions based on those memories.

## Features

- **Voice memory capture:** Receives conversation memories from Omi.
- **Persistent memory:** Stores memory text and embeddings in Qdrant.
- **Semantic search:** Finds relevant memories based on meaning.
- **AI chat:** Uses Lyzr to answer questions with relevant saved memories.

## Architecture

1. Omi sends conversation data to the MemoMind AI backend.
2. FastAPI extracts useful text and generates embeddings.
3. Qdrant stores the memories and supports semantic retrieval.
4. The chat endpoint retrieves relevant memories and sends them as context to Lyzr.
5. Lyzr generates an answer that is returned to the user.

## Tech Stack

- Python
- FastAPI
- Omi
- Qdrant
- Lyzr AI
- Sentence Transformers

## Local Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file using `.env.example` as a template.
5. Add your own Qdrant and Lyzr credentials to `.env`.
6. Start the server:

   `python -m uvicorn main:app --reload`

7. Open `http://127.0.0.1:8000` in your browser.

## Environment Variables

See `.env.example` for the required environment variable names. Never commit real API keys or credentials.

## Project Status

The Omi-to-Qdrant memory-saving flow, semantic search, and Lyzr-powered chat have been tested locally. Public deployment and the final demo remain to be completed.

## License

MIT