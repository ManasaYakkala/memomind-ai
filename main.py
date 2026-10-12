
import os
import uuid
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Body, HTTPException
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from pathlib import Path
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI(title="MemoMind AI")

COLLECTION_NAME = "memomind_memories"
VECTOR_SIZE = 384

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
lyzr_api_key = os.getenv("LYZR_API_KEY")

client = None
model = None


@app.on_event("startup")
def startup():
    global client, model

    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Download and load a small embedding model on first run.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="user_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print("User ID index is ready!")

    print("MemoMind AI connected to Qdrant!")


@app.get("/")
def home():
    return FileResponse(
        Path(__file__).parent / "frontend" / "index.html"
    )

@app.get("/health")
def health():
    if client is None:
        raise HTTPException(status_code=503, detail="Qdrant not connected")

    return {
        "status": "healthy",
        "qdrant_connected": True,
        "collection": COLLECTION_NAME,
    }


@app.post("/webhook")
async def receive_memory(
    request: Request,
    uid: str = "test-user",
    memory: dict = Body(...),
):
    if client is None or model is None:
        raise HTTPException(status_code=503, detail="Service is not ready")

    print("\n--- WEBHOOK PAYLOAD DEBUG ---")
    print(memory)
    structured = memory.get("structured") or {}
    segments = memory.get("transcript_segments") or []

    text_parts = [
        str(structured.get("title") or ""),
        str(structured.get("overview") or ""),
    ]

    for segment in segments:
        if isinstance(segment, dict):
            text_parts.append(str(segment.get("text") or ""))

    memory_text = "\n".join(part for part in text_parts if part).strip()

    if not memory_text:
        raise HTTPException(
            status_code=400,
            detail="No usable text found in memory payload",
        )

    vector = model.encode(memory_text).tolist()
    memory_id = str(memory.get("id") or uuid.uuid4())

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{uid}:{memory_id}")),
                vector=vector,
                payload={
                    "user_id": uid,
                    "memory_id": memory_id,
                    "text": memory_text,
                    "title": str(structured.get("title") or "Untitled memory"),
                    "category": str(structured.get("category") or "general"),
                },
            )
        ],
        wait=True,
    )

    return {
        "status": "success",
        "message": "Memory saved to Qdrant",
        "memory_id": memory_id,
    }


@app.get("/search")
def search_memories(q: str, uid: str =  "u0xzc2oheMWNCkd0SRMMhVgDaU23"):
    if client is None or model is None:
        raise HTTPException(status_code=503, detail="Service is not ready")

    query_vector = model.encode(q).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=uid),
                )
            ]
        ),
        limit=5,
        with_payload=True,
    ).points

    return {
        "query": q,
        "results": [
            {
                "text": point.payload.get("text", ""),
                "title": point.payload.get("title", ""),
                "score": point.score,
            }
            for point in results
        ],
    }


@app.post("/chat")
def chat_with_lyzr(
    message: str = Body(..., embed=True),
    uid: str = "u0xzc2oheMWNCkd0SRMMhVgDaU23",
):
    if not lyzr_api_key:
        raise HTTPException(
            status_code=500,
            detail="LYZR_API_KEY is missing from .env",
        )

    agent_id = os.getenv("LYZR_AGENT_ID")
    lyzr_user_id = os.getenv("LYZR_USER_ID")

    if not agent_id or not lyzr_user_id:
        raise HTTPException(
            status_code=500,
            detail="LYZR_AGENT_ID or LYZR_USER_ID is missing from .env",
        )

    if client is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Memory service is not ready",
        )

    # Find memories belonging to this website user.
    query_vector = model.encode(message).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=uid),
                )
            ]
        ),
        limit=5,
        with_payload=True,
    ).points

    memory_context = "\n\n".join(
        str(point.payload.get("text", ""))
        for point in results
        if point.payload and point.payload.get("text")
    )
    

    if memory_context:
        prompt = (
            "Answer the user's question using the relevant personal memories "
            "below when helpful. Do not invent memories. If the memories do "
            "not answer the question, say so honestly.\n\n"
            f"Relevant saved memories:\n{memory_context}\n\n"
            f"User's question: {message}"
        )
    else:
        prompt = (
            "No relevant saved memories were found. Do not claim to remember "
            "personal facts that were not provided. Answer honestly.\n\n"
            f"User's question: {message}"
        )

    session_id = f"{agent_id}-memomind-session"

    try:
        response = requests.post(
            "https://agent-prod.studio.lyzr.ai/v3/inference/chat/",
            headers={
                "x-api-key": lyzr_api_key,
                "Content-Type": "application/json",
            },
            json={
                "user_id": lyzr_user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "message": prompt,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        if exc.response is not None:
            error_detail = (
                f"Lyzr returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:1000]}"
            )
        else:
            error_detail = str(exc)

        raise HTTPException(status_code=502, detail=error_detail)


    agent_id = os.getenv("LYZR_AGENT_ID")
    user_id = os.getenv("LYZR_USER_ID")

    if not agent_id or not user_id:
        raise HTTPException(
            status_code=500,
            detail="LYZR_AGENT_ID or LYZR_USER_ID is missing from .env",
        )

    session_id = f"{agent_id}-memomind-session"

    try:
        response = requests.post(
            "https://agent-prod.studio.lyzr.ai/v3/inference/chat/",
            headers={
                "x-api-key": lyzr_api_key,
                "Content-Type": "application/json",
            },
            json={
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "message": message,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    
    except requests.RequestException as exc:
        if exc.response is not None:
            error_detail = (
                f"Lyzr returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:1000]}"
            )
        else:
            error_detail = str(exc)

        raise HTTPException(
            status_code=502,
            detail=error_detail,
        )


@app.post("/omi/conversation")
async def omi_conversation(
    request: Request,
    uid: str = "omi-user",
    payload: dict = Body(...),
):
    return await receive_memory(
        request=request,
        uid=uid,
        memory=payload,
    )
