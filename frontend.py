import os
import hashlib
import json
import psycopg2
import numpy as np
from langchain_community.llms import Ollama
from fastapi import FastAPI, HTTPException, Depends, UploadFile, Form, Request, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi.background import BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Union, List
import uuid
from jose import JWTError, jwt
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import hmac
import threading
from dotenv import load_dotenv

# ------------------------------------------------------------------------
# IMPORTANT: import your factory method from the storage_integration script
# ------------------------------------------------------------------------
from storage_integrations import integrate_data_into_datalake

# Import the improved backend module (with multi-vector embeddings, etc.)
import backend

# For summarizing long conversations (optional huggingface approach)
try:
    from transformers import pipeline
    conversation_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except:
    conversation_summarizer = None

# Evaluate library for text metrics
import evaluate
import math

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# PostgreSQL connection
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Slack Bot configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Locks for concurrency
model_lock = threading.Lock()
llm_lock = threading.Lock()

# Initialize backend (Milvus, Neo4j, Models)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testtest")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

backend.initialize_all(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    milvus_host=MILVUS_HOST,
    milvus_port=MILVUS_PORT
)

# Initialize LLM (you can choose any model in Ollama)
llm = Ollama(model="phi4:latest")

################################################################################
# Configurable Constants
################################################################################

MAX_CONVERSATION_CHARS = 3000
MAX_CONTEXT_CHARS = 3000

# If conversation grows beyond this length, we do an automatic summary
CONVERSATION_SUMMARY_TRIGGER = 4000

################################################################################
# Pydantic Models
################################################################################

class UserCredentials(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    new_chat: Optional[bool] = True
    chat_id: Optional[str] = None

class RegistrationResponse(BaseModel):
    message: str

class LoginResponse(BaseModel):
    access_token: str
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[str]
    chat_id: Optional[str]

class UpdateRoleRequest(BaseModel):
    username: str
    new_role: str

class CreateWorkspaceRequest(BaseModel):
    name: str

class AssignUserRequest(BaseModel):
    user_id: int

class StorageConfig(BaseModel):
    datalake_type: str
    config: dict

################################################################################
# Auth / Helpers
################################################################################

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return int(user_id)
    except JWTError:
        raise credentials_exception

async def get_current_user_with_role(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise credentials_exception

        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SELECT id, role, workspace_id FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if not result:
            raise credentials_exception
        return {"user_id": result[0], "role": result[1], "workspace_id": result[2]}
    except JWTError:
        raise credentials_exception

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def save_conversation(user_id, chat_id, query, response):
    """
    Saves user+assistant messages in the DB.
    We store them in user_conversations as raw text.
    Potentially used later for summarization if it becomes too large.
    """
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        sanitized_query = query.encode("utf-8", "replace").decode("utf-8")
        sanitized_response = response.encode("utf-8", "replace").decode("utf-8")
        cursor.execute(
            """
            INSERT INTO user_conversations (user_id, chat_id, conversation)
            VALUES (%s, %s, %s)
            """,
            (user_id, chat_id, f"User: {sanitized_query}\nAI: {sanitized_response}")
        )
        connection.commit()
    except Exception as e:
        print(f"Error saving conversation: {e}")
    finally:
        cursor.close()
        connection.close()

def role_required(required_roles: list):
    def decorator(current_user=Depends(get_current_user_with_role)):
        if current_user["role"] not in required_roles:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        return current_user
    return decorator

################################################################################
# Query Refinement & Summaries
################################################################################

def refine_query(original_query: str, conversation_context: str) -> str:
    """
    Use the LLM to produce a short, direct refined query in the same language as the user.
    """
    prompt_for_refinement = f"""
        You are a query refiner. Given the conversation so far and the user's latest query,
        rewrite the latest query into a short, direct query that will best match relevant documents
        in the knowledge base. If the conversation is in Czech, keep it in Czech; if in English,
        keep it in English. Avoid any extra explanation or commentary. Simply return the refined query that has all important information.

        Conversation so far:
        {conversation_context}

        User's latest query: {original_query}

        Refined query (no additional text, just the query):
        """.strip()

    with llm_lock:
        refined_query = llm.invoke(prompt_for_refinement).strip()
    return refined_query

def summarize_conversation(conversation_text: str) -> str:
    """
    Summarizes the conversation if conversation_summarizer is available;
    otherwise, do a naive approach.
    """
    if conversation_summarizer:
        try:
            result = conversation_summarizer(conversation_text, max_length=100, min_length=50, do_sample=False)
            summary = result[0]["summary_text"]
            return summary.strip()
        except Exception as e:
            print(f"Summarization error: {e}")

    # fallback naive approach
    truncated = conversation_text[:500] + "..."
    return f"Summary of conversation: {truncated}"

def maybe_summarize_long_conversation(user_id: int, chat_id: str):
    """
    If the conversation is too long, we automatically summarize it
    and store that summary as a new conversation entry (like a running memory).
    """
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation 
            FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (user_id, chat_id)
        )
        rows = cursor.fetchall()
        full_conv = "\n".join([r[0] for r in rows])
        if len(full_conv) > CONVERSATION_SUMMARY_TRIGGER:
            # Summarize
            summary = summarize_conversation(full_conv)
            # Insert the summary as a new "turn"
            cursor.execute(
                """
                INSERT INTO user_conversations (user_id, chat_id, conversation)
                VALUES (%s, %s, %s)
                """,
                (user_id, chat_id, f"AI (conversation summary): {summary}")
            )
            connection.commit()
            return True
        return False
    except Exception as e:
        print(f"Error in maybe_summarize_long_conversation: {e}")
    finally:
        cursor.close()
        connection.close()

################################################################################
# Multi-Vector + Graph Retrieval
################################################################################

def hybrid_search(query: str, top_k: int = 5) -> list:
    """
    Demonstrates a multi-vector retrieval (semantic + lexical) approach.
    1) Embed the query using both models
    2) Search in both Milvus collections
    3) Merge results (by average or max of similarity)
    4) Return top_k doc_ids
    """
    # Step 1: embed query in both models
    with model_lock:
        sem_emb = backend.semantic_embedding_model.encode([query], show_progress_bar=False)[0].astype(np.float32)
        lex_emb = backend.lexical_embedding_model.encode([query], show_progress_bar=False)[0].astype(np.float32)

    # Step 2: do Milvus searches
    sem_search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    lex_search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    sem_results = backend.semantic_collection.search(
        data=[sem_emb.tolist()],
        anns_field="embedding",
        param=sem_search_params,
        limit=top_k * 2,
        output_fields=["document_id"]
    )[0]

    lex_results = backend.lexical_collection.search(
        data=[lex_emb.tolist()],
        anns_field="embedding",
        param=lex_search_params,
        limit=top_k * 2,
        output_fields=["document_id"]
    )[0]

    # Step 3: unify and rank
    score_map = {}

    for hit in sem_results:
        doc_id = hit.entity.get("document_id")
        score = hit.score
        if doc_id not in score_map:
            score_map[doc_id] = []
        score_map[doc_id].append(score)

    for hit in lex_results:
        doc_id = hit.entity.get("document_id")
        score = hit.score
        if doc_id not in score_map:
            score_map[doc_id] = []
        score_map[doc_id].append(score)

    doc_id_scores = []
    for doc_id, scores in score_map.items():
        avg_score = sum(scores) / len(scores)
        doc_id_scores.append((doc_id, avg_score))

    doc_id_scores.sort(key=lambda x: x[1], reverse=True)
    top_doc_ids = [t[0] for t in doc_id_scores[:top_k]]
    return top_doc_ids

def generate_cypher_query(refined_query: str) -> str:
    """
    Use LLM to generate a possible Cypher query to find relevant docs in the graph
    ...
    """
    prompt = f"""
    You are a Cypher query generator. The user (in the knowledge base) asked a refined query:
    '{refined_query}'

    We have a Neo4j graph with :Document, :Topic, :Entity, and relationships like:
    (Document)-[:HAS_TOPIC]->(Topic), (Document)-[:MENTIONS]->(Entity),
    (Document)-[:RELATED {{type: 'SIMILAR_TO', ...}}]->(Document).

    Generate a short Cypher query that tries to find Document nodes relevant to the user query.
    Only output the Cypher.
    """.strip()

    with llm_lock:
        possible_cypher = llm.invoke(prompt).strip()
    return possible_cypher

def run_cypher_query(query_text: str, top_k: int = 5) -> list:
    """
    Attempts to run a given Cypher query, expecting it to return a list of doc_ids
    from Document nodes. We'll parse them out.
    """
    doc_ids = []
    with backend.driver.session() as session:
        try:
            result = session.run(query_text)
            for rec in result:
                if "doc_id" in rec.keys():
                    doc_ids.append(rec["doc_id"])
                else:
                    val = rec.values()[0]
                    if isinstance(val, int):
                        doc_ids.append(val)
        except Exception as e:
            print(f"Cypher query failed or invalid: {e}")
    doc_ids = list(set(doc_ids))
    return doc_ids[:top_k]

def retrieve_docs_from_neo4j(doc_ids: list) -> list:
    """
    Given a list of doc_ids, fetch their content and metadata from Neo4j.
    """
    if not doc_ids:
        return []

    with backend.driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            WHERE d.doc_id IN $doc_ids
            RETURN d.content AS content, d.metadata AS metadata
            """,
            doc_ids=doc_ids
        )
        documents = []
        for record in result:
            meta = json.loads(record['metadata'])
            documents.append({
                "content": record["content"],
                "filename": meta.get("filename", "Unknown")
            })
    return documents

def get_hybrid_plus_cypher_docs(refined_query: str, top_k: int = 5, use_cypher_expansion: bool = True) -> list:
    """
    1. Multi-vector search in Milvus
    2. (Optional) Generate a Cypher query to find relevant docs in Neo4j
    3. Merge doc_ids, retrieve content from Neo4j
    """
    doc_ids_hybrid = hybrid_search(refined_query, top_k=top_k)

    doc_ids_cypher = []
    if use_cypher_expansion:
        possible_cypher = generate_cypher_query(refined_query)
        doc_ids_cypher = run_cypher_query(possible_cypher, top_k=top_k)

    all_ids = list(set(doc_ids_hybrid + doc_ids_cypher))
    docs = retrieve_docs_from_neo4j(all_ids)
    return docs

################################################################################
# Slack Helpers
################################################################################

async def verify_slack_signature(request: Request):
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    if abs(int(timestamp) - int(datetime.now().timestamp()) ) > 60 * 5:
        return False

    request_body = await request.body()
    sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    computed_signature = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()

    slack_signature = request.headers.get("X-Slack-Signature")
    return hmac.compare_digest(computed_signature, slack_signature)

async def process_slack_command(user_query: str, channel_id: str):
    try:
        response = generate_response(
            QueryRequest(query=user_query, new_chat=True),
            current_user={"user_id": 1, "role": "superadmin", "workspace_id": None}
        )
        slack_client.chat_postMessage(channel=channel_id, text=response.response)
    except Exception as e:
        print(f"Error processing Slack command: {e}")
        slack_client.chat_postMessage(
            channel=channel_id,
            text="Sorry, something went wrong while processing your request."
        )

def get_storage_settings():
    config_path = "storage_config.json"
    if not os.path.exists(config_path):
        raise HTTPException(status_code=500, detail="Storage configuration not found.")
    with open(config_path, "r") as f:
        return json.load(f)

################################################################################
# Existing Endpoints
################################################################################

@app.get("/chats", response_model=dict)
def get_user_chats(current_user_id: int = Depends(get_current_user)):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT chat_id, MAX(conversation) AS latest_message
            FROM user_conversations
            WHERE user_id = %s
            GROUP BY chat_id
            ORDER BY MAX(id) DESC
            """,
            (current_user_id,)
        )
        result = cursor.fetchall()
        chats = [{"chat_id": row[0], "latest_message": row[1]} for row in result]
        return {"chats": chats}
    finally:
        cursor.close()
        connection.close()

@app.get("/chat/history/{chat_id}", response_model=dict)
def get_chat_history(chat_id: str, current_user_id: int = Depends(get_current_user)):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (current_user_id, chat_id)
        )
        result = cursor.fetchall()
        history = []
        for record in result:
            conversations = record[0].split("\n")
            history.extend(conversations)
        return {"history": history}
    finally:
        cursor.close()
        connection.close()

@app.post("/register")
def register_user(username: str = Form(...), password: str = Form(...)):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        connection.commit()
        return {"message": "Registration successful"}
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        cursor.close()
        connection.close()

@app.post("/login")
def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, role FROM users WHERE username = %s AND password = %s", (username, hashed_password))
        result = cursor.fetchone()
        if result:
            user_id, user_role = result
            access_token = create_access_token(data={"sub": str(user_id)})
            return {"access_token": access_token, "message": "Login successful", "role": user_role}
        raise HTTPException(status_code=401, detail="Invalid username or password")
    finally:
        cursor.close()
        connection.close()

@app.post("/workspaces")
def create_workspace(
    request: CreateWorkspaceRequest,
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO workspaces (name) VALUES (%s) RETURNING id", (request.name,))
        workspace_id = cursor.fetchone()[0]
        connection.commit()
        return {"message": "Workspace created", "workspace_id": workspace_id}
    finally:
        cursor.close()
        connection.close()

@app.post("/workspaces/{workspace_id}/assign-user")
def assign_user_to_workspace(
    workspace_id: int,
    request: AssignUserRequest,
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO user_workspaces (user_id, workspace_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (request.user_id, workspace_id)
        )
        connection.commit()
        return {"message": f"User {request.user_id} assigned to workspace {workspace_id}"}
    finally:
        cursor.close()
        connection.close()

@app.post("/documents")
def upload_document(
    file: UploadFile = File(...),
    scope: str = Form(...),
    chat_id: Optional[str] = Form(None),
    current_user=Depends(role_required(["user", "admin", "superadmin"]))
):
    if scope not in ["chat", "profile", "workspace", "system"]:
        raise HTTPException(status_code=400, detail="Invalid scope")

    if scope == "chat" and not chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required for chat scope")

    if scope == "workspace" and current_user["role"] not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Admins and Superadmins only")
    if scope == "system" and current_user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Superadmins only")

    file_content = file.file.read().decode("utf-8")
    doc_id = hashlib.sha256(file_content.encode()).hexdigest()
    metadata = {"filename": file.filename, "scope": scope}

    # Store doc in Neo4j (omitting workspace logic for brevity)
    with backend.driver.session() as session:
        session.run(
            """
            CREATE (d:Document {
                doc_id: $doc_id,
                content: $content,
                metadata: $metadata,
                is_global: false,
                workspace_id: null
            })
            """,
            doc_id=doc_id,
            content=file_content,
            metadata=json.dumps(metadata)
        )

    return {"message": f"Document uploaded successfully with scope {scope}"}

@app.post("/chat", response_model=ChatResponse)
def generate_response(
    request: QueryRequest,
    current_user=Depends(get_current_user_with_role)
):
    """
    Main chat endpoint with multi-vector retrieval + optional graph expansions,
    conversation summarization, and query refinement.
    """
    print(f"Request received: {request.dict()}")

    ########################################################################
    # LLM-BASED MODERATION CHECK (ADDED CODE)
    ########################################################################
    if not llm_moderation_check(request.query):
        raise HTTPException(
            status_code=403,
            detail="Your query is disallowed by the moderation policy."
        )
    ########################################################################

    # 1) Get or create chat_id
    if request.new_chat:
        chat_id = str(uuid.uuid4())
        conversation_so_far = ""
    else:
        if not request.chat_id:
            raise HTTPException(status_code=400, detail="chat_id is required when new_chat is False")
        chat_id = request.chat_id

        connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT conversation 
                FROM user_conversations 
                WHERE user_id = %s AND chat_id = %s 
                ORDER BY id ASC
                """,
                (current_user["user_id"], chat_id)
            )
            rows = cursor.fetchall()
            conversation_so_far = "\n".join([r[0] for r in rows])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")
        finally:
            cursor.close()
            connection.close()

    # 2) Possibly summarize
    maybe_summarize_long_conversation(current_user["user_id"], chat_id)

    # Reload if summary was added
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT conversation 
            FROM user_conversations 
            WHERE user_id = %s AND chat_id = %s 
            ORDER BY id ASC
            """,
            (current_user["user_id"], chat_id)
        )
        rows = cursor.fetchall()
        conversation_so_far = "\n".join([r[0] for r in rows])
    finally:
        cursor.close()
        connection.close()

    # 3) Refine query
    refined_query = refine_query(request.query, conversation_so_far)
    print(f"Refined query: {refined_query}")

    # 4) Retrieve docs
    docs = get_hybrid_plus_cypher_docs(refined_query, top_k=3, use_cypher_expansion=True)

    # Build short context
    context_text = ""
    for doc in docs:
        piece = f"{doc['content']}\n(Source: {doc['filename']})\n\n"
        if len(context_text + piece) < MAX_CONTEXT_CHARS:
            context_text += piece
        else:
            context_text += "... [truncated]"
            break

    truncated_conversation = conversation_so_far
    if len(truncated_conversation) > MAX_CONVERSATION_CHARS:
        truncated_conversation = truncated_conversation[:MAX_CONVERSATION_CHARS] + " ... [truncated]"

    # 5) Construct prompt
    prompt = f"""
        You are a helpful assistant who provides concise, step-by-step solutions.
        Conversation so far:
        {truncated_conversation}

        Relevant context:
        {context_text}

        Question:
        {request.query}

        Your concise answer:
    """.strip()

    # 6) LLM response
    with llm_lock:
        response_text = llm.invoke(prompt).strip()
    response_text = response_text.encode('utf-8', errors='replace').decode('utf-8')

    # 7) Save turn
    save_conversation(current_user["user_id"], chat_id, request.query, response_text)

    # Return doc filenames
    source_names = [doc['filename'] for doc in docs]
    sources_str = "\n".join(source_names)
    final_answer = f"{response_text}\n\nSources:\n{sources_str}"

    return {
        "response": final_answer,
        "sources": ", ".join(source_names),
        "chat_id": chat_id
    }

@app.post("/slack/events")
async def slack_events(request: Request):
    if not await verify_slack_signature(request):
        return JSONResponse(status_code=403, content={"message": "Invalid signature"})

    body = await request.json()
    event = body.get("event", {})
    if event.get("type") == "message" and not event.get("bot_id"):
        user_query = event.get("text")
        channel_id = event.get("channel")
        try:
            response = generate_response(
                QueryRequest(query=user_query, new_chat=True),
                current_user={"user_id": 1, "role": "superadmin", "workspace_id": None}
            )
            slack_client.chat_postMessage(channel=channel_id, text=response.response)
        except SlackApiError as e:
            print(f"Error sending message: {e.response['error']}")

    return JSONResponse(content={"message": "Event received"})

@app.post("/slack/command")
async def slack_command(request: Request, background_tasks: BackgroundTasks):
    if not await verify_slack_signature(request):
        return JSONResponse(status_code=403, content={"message": "Invalid signature"})

    form_data = await request.form()
    user_query = form_data.get("text")
    channel_id = form_data.get("channel_id")
    background_tasks.add_task(process_slack_command, user_query, channel_id)
    return JSONResponse(content={"response_type": "ephemeral", "text": "Processing your request..."})

@app.post("/update-role")
def update_role(
    request: UpdateRoleRequest,
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    if request.new_role not in ["user", "admin", "superadmin"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET role = %s WHERE username = %s RETURNING id",
            (request.new_role, request.username)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        return {"message": f"Role updated for user {request.username} to {request.new_role}"}
    finally:
        cursor.close()
        connection.close()

@app.get("/admin/users")
def get_all_users(current_user=Depends(role_required(["superadmin"]))):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, username, role FROM users ORDER BY id ASC")
        result = cursor.fetchall()
        users = [{"id": row[0], "username": row[1], "role": row[2]} for row in result]
        return {"users": users}
    finally:
        cursor.close()
        connection.close()

@app.post("/admin/users/{user_id}/change-username")
def change_username(
    user_id: int,
    new_username: str = Form(...),
    current_user=Depends(role_required(["superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET username = %s WHERE id = %s RETURNING id",
            (new_username, user_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        return {"message": "Username updated successfully"}
    finally:
        cursor.close()
        connection.close()

@app.post("/admin/users/{user_id}/change-password")
def change_password(
    user_id: int,
    new_password: str = Form(...),
    current_user=Depends(role_required(["superadmin"]))
):
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            "UPDATE users SET password = %s WHERE id = %s RETURNING id",
            (hashed_password, user_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        connection.commit()
        return {"message": "Password updated successfully"}
    finally:
        cursor.close()
        connection.close()

@app.get("/admin/users/{user_id}/chats")
def get_user_chats_admin(
    user_id: int,
    current_user=Depends(role_required(["superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT chat_id, MAX(conversation) AS latest_message
            FROM user_conversations
            WHERE user_id = %s
            GROUP BY chat_id
            ORDER BY MAX(id) DESC
            """,
            (user_id,)
        )
        result = cursor.fetchall()
        chats = [{"chat_id": row[0], "latest_message": row[1]} for row in result]
        return {"chats": chats}
    finally:
        cursor.close()
        connection.close()

@app.post("/embed-documents")
def embed_documents(
    directory: str = Form(...),
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="Invalid directory")

    try:
        backend.process_documents(directory)
        return {"message": f"Documents in {directory} processed and embedded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspaces/{user_id}/list")
def get_user_workspaces(
    user_id: int,
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT w.id, w.name
            FROM workspaces w
            JOIN user_workspaces uw ON w.id = uw.workspace_id
            WHERE uw.user_id = %s
            """,
            (user_id,)
        )
        result = cursor.fetchall()
        workspaces = [{"workspace_id": row[0], "name": row[1]} for row in result]
        return {"workspaces": workspaces}
    finally:
        cursor.close()
        connection.close()

@app.post("/configure-storage")
def configure_storage(
    storage_config: StorageConfig,
    current_user=Depends(role_required(["superadmin"]))):
    config_path = "storage_config.json"
    with open(config_path, "w") as f:
        json.dump(storage_config.dict(), f)
    return {"message": f"Storage configured successfully for {storage_config.datalake_type}"}

@app.post("/local-datalake/upload-file")
def local_datalake_upload_file(
    file: UploadFile = File(...),
    is_global: bool = Form(False),
    workspace_id: Optional[int] = Form(None),
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    """
    Endpoint to upload a file from the user's computer into the local datalake,
    tag it with is_global or workspace_id, and embed it using backend's logic.
    """

    # 1) Read file into bytes
    filename = file.filename
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty or unreadable.")

    # 2) Store in local datalake with metadata (is_global, workspace_id, etc.)
    from storage_integrations import get_datalake
    local_dlake = get_datalake("local")  # specifically want local

    # We'll store uploaded files in "uploaded/" subfolder in local datalake
    local_path = f"uploaded/{filename}"

    # Create metadata
    metadata = {
        "filename": filename,
        "uploaded_by": current_user["user_id"],
        "is_global": is_global,
        "workspace_id": workspace_id,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    # Save file + metadata
    local_dlake.save_file_with_metadata(file_bytes, local_path, metadata)

    # 3) Embed it by calling the existing logic in backend
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_filepath = os.path.join(tmpdir, filename)
        # Write file bytes to a temporary file so we can reuse backend.read_file_content(...)
        with open(temp_filepath, "wb") as f_out:
            f_out.write(file_bytes)

        # Extract text from the file
        content = backend.read_file_content(temp_filepath)
        if not content:
            raise HTTPException(status_code=400, detail="Could not parse text from the file.")

        # Split into chunks (using your existing chunk function)
        chunks = backend.chunk_text_with_langchain(content, chunk_size=1000, chunk_overlap=200)

        # Create base metadata for the Document nodes
        file_level_meta = {
            "filename": filename,
            "is_global": is_global,
            "workspace_id": workspace_id,
            "size": len(file_bytes),
            "word_count": len(content.split())
        }

        # We'll gather doc_ids + embeddings for doc-doc similarity
        doc_ids = []
        sem_embeddings = []

        for ctext in chunks:
            # This calls the same logic that:
            #   - Summarizes chunk
            #   - Creates doc node in Neo4j
            #   - Possibly does entity extraction
            #   - Stores embeddings in Milvus
            doc_id = backend.process_chunk(ctext, file_level_meta)
            doc_ids.append(doc_id)

            # For doc-doc similarity, store the semantic embedding
            sem_emb = backend.semantic_embedding_model.encode([ctext], show_progress_bar=False)[0]
            sem_embeddings.append(sem_emb)

        # Finally, link these new chunks to each other if they're similar
        backend.compute_batch_similarities(doc_ids, sem_embeddings, threshold=0.7)

    return {
        "message": f"File '{filename}' uploaded and embedded successfully.",
        "is_global": is_global,
        "workspace_id": workspace_id,
        "local_path": local_path
    }

# ----------------------------------------------------------------------------
# NEW ENDPOINTS (do not modify anything above)
# ----------------------------------------------------------------------------
from fastapi import Form

@app.post("/admin/copy-google-drive-to-local")
def copy_google_drive_to_local(
    folder_id: str = Form(...),
    is_global: bool = Form(False),
    current_user=Depends(role_required(["superadmin"]))
):
    """
    Copy files from Google Drive into the local datalake folder,
    then set is_global in their metadata.
    We call integrate_data_into_datalake("google_drive", "local", folder_id=...),
    then update .metadata.json files to reflect is_global.
    """
    try:
        integrate_data_into_datalake(
            provider="google_drive",
            datalake_type="local",
            folder_id=folder_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    base_path = os.environ.get("LOCAL_DATALAKE_PATH", "local_datalake")
    google_drive_dir = os.path.join(base_path, "google_drive")

    for root, dirs, files in os.walk(google_drive_dir):
        for filename in files:
            if filename.endswith(".metadata.json"):
                meta_path = os.path.join(root, filename)
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata["is_global"] = is_global
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception as ex:
                    print(f"Could not update metadata file {meta_path}: {ex}")

    return {
        "message": "Successfully copied files from Google Drive to local datalake",
        "is_global": is_global
    }

@app.post("/admin/configure-storage-dashboard")
def configure_storage_dashboard(
    datalake_type: str = Form(...),
    current_user=Depends(role_required(["superadmin"]))
):
    """
    Minimal wrapper around /configure-storage for a simpler Admin UI
    that only sets the datalake type and leaves config empty or default.
    """
    dummy_config = {}
    payload = StorageConfig(datalake_type=datalake_type, config=dummy_config)

    config_path = "storage_config.json"
    with open(config_path, "w") as f:
        json.dump(payload.dict(), f)

    return {"message": f"Storage configured successfully for {datalake_type}"}

# ----------------------------------------------------------------------------
# LLM-BASED MODERATION HELPERS (ADDED CODE)
# ----------------------------------------------------------------------------

def llm_moderation_check(query: str) -> bool:
    """
    Use the LLM to decide if a query is ALLOWED or DISALLOWED based on a simple inline policy.
    Returns True if allowed, False if disallowed.
    """
    # A minimal policy prompt:
    policy_prompt = f"""
System: You are a strict content policy checker. The user input is below.
If the user is discussing or requesting disallowed topics (like politics), respond EXACTLY 'DISALLOWED'.
Otherwise respond EXACTLY 'ALLOWED'.

User input:
{query}
""".strip()

    with llm_lock:
        classification = llm.invoke(policy_prompt).strip().upper()

    # If the LLM says DISALLOWED, we return False. Otherwise True.
    if "DISALLOWED" in classification:
        return False
    return True

# ----------------------------------------------------------------------------
# EVALUATION CODE WITH TENSORBOARD
# ----------------------------------------------------------------------------

class RetrievalEvaluationItem(BaseModel):
    query: str
    ground_truth_docs: List[str]
    retrieved_docs: List[str]
    ground_truth_answer: Optional[str] = None
    system_answer: Optional[str] = None

class EvaluationRequest(BaseModel):
    data: List[RetrievalEvaluationItem]

rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")

def compute_precision_recall_f1(
    relevant: set,
    retrieved: set
):
    if not relevant and not retrieved:
        return 1.0, 1.0, 1.0
    tp = len(relevant.intersection(retrieved))
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1

def compute_mrr(relevant_docs: set, retrieved_docs: List[str]) -> float:
    for idx, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            return 1.0 / (idx + 1)
    return 0.0

def compute_dcg(relevance_list: List[int]) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_list):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg

def compute_ndcg(relevant_docs: set, retrieved_docs: List[str], k: Optional[int] = None) -> float:
    if not k:
        k = len(retrieved_docs)
    truncated = retrieved_docs[:k]
    rel_list = [1 if doc_id in relevant_docs else 0 for doc_id in truncated]
    dcg = compute_dcg(rel_list)
    ideal_rel_list = sorted(rel_list, reverse=True)
    idcg = compute_dcg(ideal_rel_list)
    if idcg == 0.0:
        return 1.0 if dcg == 0.0 else 0.0
    return dcg / idcg

def compute_text_metrics(predictions: List[str], references: List[str]):
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)
    bert_results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="bert-base-uncased"
    )

    results_summary = {}
    results_summary["rouge1_f"] = rouge_results["rouge1"]
    results_summary["rouge2_f"] = rouge_results["rouge2"]
    results_summary["rougeL_f"] = rouge_results["rougeL"]
    results_summary["meteor"] = meteor_results["meteor"]
    results_summary["bertscore_precision"] = sum(bert_results["precision"]) / len(bert_results["precision"])
    results_summary["bertscore_recall"]    = sum(bert_results["recall"]) / len(bert_results["recall"])
    results_summary["bertscore_f1"]        = sum(bert_results["f1"]) / len(bert_results["f1"])
    return results_summary

@app.post("/evaluate")
def evaluate_system(request_data: EvaluationRequest):
    """
    Accepts a list of evaluation items, each containing:
     - query
     - ground_truth_docs
     - retrieved_docs (system-provided)
     - ground_truth_answer

    For each item, this endpoint:
      1. Refines the query and retrieves documents.
      2. Constructs a prompt and generates a new system answer using the LLM.
      3. Computes retrieval metrics (using the provided retrieved_docs) and
         generative (text) metrics comparing the new answer with the ground truth.
      4. Logs all metrics to TensorBoard and returns the aggregated metrics
         along with the generated answers.
    """
    if not TENSORBOARD_AVAILABLE:
        raise HTTPException(status_code=500, detail="TensorBoard not installed or unavailable.")

    data = request_data.data

    # Create a TensorBoard SummaryWriter (using a timestamped directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/evaluation_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize accumulators for retrieval metrics
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_mrr = 0.0
    sum_ndcg = 0.0
    retrieval_count = 0

    # For generative metrics (text evaluation)
    all_predictions = []
    all_references = []
    generated_answers = []

    # Process each evaluation item
    for idx, item in enumerate(data):
        # --- Retrieval metrics are computed based on provided docs ---
        relevant_docs = set(item.ground_truth_docs)
        retrieved_docs = item.retrieved_docs
        if relevant_docs or retrieved_docs:
            p, r, f1 = compute_precision_recall_f1(relevant_docs, set(retrieved_docs))
            mrr = compute_mrr(relevant_docs, retrieved_docs)
            ndcg = compute_ndcg(relevant_docs, retrieved_docs)
            sum_precision += p
            sum_recall += r
            sum_f1 += f1
            sum_mrr += mrr
            sum_ndcg += ndcg
            retrieval_count += 1
        else:
            p = r = f1 = mrr = ndcg = 1.0

        # Log per-item retrieval metrics
        writer.add_scalar("per_item/precision", p, idx)
        writer.add_scalar("per_item/recall", r, idx)
        writer.add_scalar("per_item/f1", f1, idx)
        writer.add_scalar("per_item/mrr", mrr, idx)
        writer.add_scalar("per_item/ndcg", ndcg, idx)

        # --- Generate a new system answer ---
        # Assume an empty conversation context for evaluation.
        conversation_so_far = ""
        refined_query = refine_query(item.query, conversation_so_far)
        docs = get_hybrid_plus_cypher_docs(refined_query, top_k=3, use_cypher_expansion=True)

        # Build a context string from the retrieved documents
        context_text = ""
        for doc in docs:
            piece = f"{doc['content']}\n(Source: {doc['filename']})\n\n"
            if len(context_text + piece) < MAX_CONTEXT_CHARS:
                context_text += piece
            else:
                context_text += "... [truncated]"
                break

        # Construct the prompt using the query and context
        prompt = f"""
        You are a helpful assistant who provides concise, step-by-step solutions.
        Conversation so far: {conversation_so_far}
        Relevant context: {context_text}
        Question: {item.query}
        Your concise answer:
        """.strip()

        # Generate the new system answer using your LLM
        with llm_lock:
            generated_answer = llm.invoke(prompt).strip()
        generated_answer = generated_answer.encode('utf-8', errors='replace').decode('utf-8')

        # Fill the evaluation item with the newly generated answer
        item.system_answer = generated_answer
        generated_answers.append(generated_answer)

        # For generative metrics, accumulate ground truth and generated answer
        if item.ground_truth_answer:
            all_references.append(item.ground_truth_answer)
            all_predictions.append(generated_answer)

    # Compute aggregated retrieval metrics
    if retrieval_count > 0:
        avg_precision = sum_precision / retrieval_count
        avg_recall = sum_recall / retrieval_count
        avg_f1 = sum_f1 / retrieval_count
        avg_mrr = sum_mrr / retrieval_count
        avg_ndcg = sum_ndcg / retrieval_count
    else:
        avg_precision = avg_recall = avg_f1 = avg_mrr = avg_ndcg = 0.0

    writer.add_scalar("retrieval/precision", avg_precision, 0)
    writer.add_scalar("retrieval/recall", avg_recall, 0)
    writer.add_scalar("retrieval/f1", avg_f1, 0)
    writer.add_scalar("retrieval/mrr", avg_mrr, 0)
    writer.add_scalar("retrieval/ndcg", avg_ndcg, 0)

    # Compute generative (text) metrics if ground truth answers exist
    if all_predictions and all_references:
        generative_metrics = com    pute_text_metrics(all_predictions, all_references)
        writer.add_scalar("generative/rouge1_f", generative_metrics["rouge1_f"], 0)
        writer.add_scalar("generative/rouge2_f", generative_metrics["rouge2_f"], 0)
        writer.add_scalar("generative/rougeL_f", generative_metrics["rougeL_f"], 0)
        writer.add_scalar("generative/meteor", generative_metrics["meteor"], 0)
        writer.add_scalar("generative/bertscore_precision", generative_metrics["bertscore_precision"], 0)
        writer.add_scalar("generative/bertscore_recall", generative_metrics["bertscore_recall"], 0)
        writer.add_scalar("generative/bertscore_f1", generative_metrics["bertscore_f1"], 0)
    else:
        generative_metrics = {
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "meteor": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0
        }

    writer.close()

    return {
        "retrieval_metrics": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "mrr": avg_mrr,
            "ndcg": avg_ndcg
        },
        "generative_metrics": generative_metrics,
        "tensorboard_logdir": log_dir,
        "generated_system_answers": generated_answers
    }

# ----------------------------------------------------------------------------
# ADD THESE FOR YOUR MODERATION CONFIG
# ----------------------------------------------------------------------------

class ModerationConfig(BaseModel):
    moderation_policy_prompt: str = ""

@app.get("/admin/moderation-config")
def get_moderation_config(current_user=Depends(role_required(["superadmin"]))):
    """
    Returns JSON with the current moderation policy (prompt) from moderation_config.json.
    """
    config_path = "moderation_config.json"
    if not os.path.exists(config_path):
        return {"moderation_policy_prompt": ""}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/admin/moderation-config")
def update_moderation_config(
    new_config: ModerationConfig,
    current_user=Depends(role_required(["superadmin"]))
):
    """
    Updates moderation_config.json with new data from superadmin.
    """
    config_path = "moderation_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(new_config.dict(), f, ensure_ascii=False, indent=2)
    return {"message": "Moderation config updated successfully."}


# ----------------------------------------------------------------------------
# NEW: API KEY CREATION & LISTING FOR ADMIN/SUPERADMIN
# ----------------------------------------------------------------------------
import secrets

class ApiKeyCreateRequest(BaseModel):
    name: str  # e.g. "Popup Integration" or "Marketing Website"
    is_global: bool = True
    workspace_id: Optional[int] = None  # if you want to limit usage to a workspace, set this

@app.post("/admin/api-keys/generate")
def generate_api_key(
    req: ApiKeyCreateRequest,
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    """
    Generates a new random API key for external usage, optionally tied to a workspace or global.
    Stores it in 'api_keys' table (which you must create in DB).
    Returns the raw key_value once (make sure to copy it!).
    """

    # Make a random hex token
    new_key_value = secrets.token_hex(32)  # e.g. 64-char hex string

    # Insert into a hypothetical 'api_keys' table.
    # You must create a table in Postgres like:
    # CREATE TABLE IF NOT EXISTS api_keys (
    #   id SERIAL PRIMARY KEY,
    #   name TEXT,
    #   key_value TEXT UNIQUE,
    #   is_global BOOLEAN DEFAULT TRUE,
    #   workspace_id INT,
    #   created_by INT,
    #   created_at TIMESTAMP DEFAULT NOW()
    # );
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO api_keys (name, key_value, is_global, workspace_id, created_by)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (req.name, new_key_value, req.is_global, req.workspace_id, current_user["user_id"])
        )
        api_key_id = cursor.fetchone()[0]
        connection.commit()
    except Exception as e:
        cursor.close()
        connection.close()
        raise HTTPException(status_code=500, detail=f"Error creating API key: {e}")
    finally:
        cursor.close()
        connection.close()

    return {
        "api_key_id": api_key_id,
        "api_key_value": new_key_value,
        "is_global": req.is_global,
        "workspace_id": req.workspace_id,
        "message": "API Key generated successfully. Please copy the api_key_value now, as it won't be shown again."
    }

@app.get("/admin/api-keys")
def list_api_keys(current_user=Depends(role_required(["admin", "superadmin"]))):
    """
    Lists existing API keys from the 'api_keys' table.
    - If role == 'superadmin', show all keys.
    - If role == 'admin', show only keys created_by this user.
    For security reasons, consider hiding the raw key_value in production.
    """
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        # If superadmin, see all keys
        if current_user["role"] == "superadmin":
            query = """
                SELECT id, name, key_value, is_global, workspace_id, created_by, created_at
                FROM api_keys
                ORDER BY id ASC
            """
            cursor.execute(query)
        else:
            # Admin: see only your own keys
            query = """
                SELECT id, name, key_value, is_global, workspace_id, created_by, created_at
                FROM api_keys
                WHERE created_by = %s
                ORDER BY id ASC
            """
            cursor.execute(query, (current_user["user_id"],))

        rows = cursor.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "name": r[1],
                "key_value": r[2],
                "is_global": r[3],
                "workspace_id": r[4],
                "created_by": r[5],
                "created_at": str(r[6])
            })
        return {"api_keys": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        connection.close()
from fastapi import Header

@app.post("/external-chat", response_model=ChatResponse)
def external_chat(
    request: QueryRequest,
    x_api_key: str = Header(None)
):
    """
    Minimal external endpoint that reuses the /chat logic.
    1) We validate the API key from 'api_keys' table.
    2) If valid, we pass a 'fake' user object with user_id=0 into the existing /chat function.
    3) The /chat logic is reused exactly.
    """

    # 1) Check if x_api_key is present
    if not x_api_key:
        raise HTTPException(status_code=403, detail="Missing X-Api-Key header.")

    # 2) Validate the API key from your database
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        cursor.execute("""
            SELECT id, is_global, workspace_id
            FROM api_keys
            WHERE key_value = %s
            LIMIT 1
        """, (x_api_key,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=403, detail="Invalid or unknown API key.")

        # If you want to enforce is_global/workspace here, do so
        # e.g. is_global = row[1], workspace_id = row[2]
    except Exception as e:
        cursor.close()
        connection.close()
        raise HTTPException(status_code=500, detail=f"Error checking API key: {e}")
    finally:
        cursor.close()
        connection.close()

    # 3) Create a "fake" user dict with user_id=0, so the /chat logic sees "someone"
    #    You can pick any role that passes your internal checks. Typically "admin" or "superadmin."
    fake_user = {"user_id": 0, "role": "user", "workspace_id": None}

    # 4) Reuse the existing generate_response(...) function
    #    because it takes (request, current_user=...).
    #    The "Depends(get_current_user_with_role)" will be skipped when we call it directly.
    return generate_response(request, fake_user)

@app.delete("/admin/api-keys/{id}/revoke")
def revoke_api_key(
    id: int,
    current_user=Depends(role_required(["admin", "superadmin"]))
):
    """
    Revokes (deletes) the API key with the given ID from the api_keys table.
    - If role == 'superadmin', can revoke any key.
    - If role == 'admin', can only revoke keys they created themselves.
    """
    connection = psycopg2.connect(**DB_CONFIG, options='-c client_encoding=UTF8')
    cursor = connection.cursor()
    try:
        # 1) Check if the key exists
        cursor.execute("""
            SELECT id, created_by
            FROM api_keys
            WHERE id = %s
        """, (id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="API key not found")

        key_id, created_by = row

        # 2) If the current user is 'admin', ensure they own this key
        if current_user["role"] == "admin" and created_by != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="You do not have permission to revoke this key")

        # 3) Perform the delete
        cursor.execute("DELETE FROM api_keys WHERE id = %s", (key_id,))
        connection.commit()

        return {"message": f"API key {id} revoked successfully"}
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        connection.close()
