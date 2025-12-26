# ====================================== IMPORTING LIBRARIES =========================================
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
import os
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, wait_exponential
from litellm import completion

# ==================================== CREDENTIALS =========================================
load_dotenv(override=True)

# gemini_api_key = os.getenv("GEMINI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_url = os.getenv("OPENROUTER_URL")

openrouter = OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)
# 
DB_NAME = "preprocessed_db"
KNOWLEDGE_BASE_PATH = Path("knowledge-base")

collection_name = "docs"
embedding_model = "all-MiniLM-L6-v2"

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 20
FINAL_K = 10

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# this is a way that makes our code be retried after failing!
wait = wait_exponential(multiplier=1, min=10, max=240)

class Result(BaseModel):
    page_content: str
    metadata: dict
# ========================================================================================

# DEFINING DATA TEMPLATE FOR RANKING OF CHUNKS:
class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

# DEFINING FUNCTION TO RANK THE CHUNKS:
#@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge
base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by 
relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk
first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are 
provided 
with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = openrouter.chat.completions.create(
        model="gpt-oss-120b",
        messages=messages,
        temperature=0,
        max_tokens=500
    )
    content = response.choices[0].message.content.strip()
    if not content.startswith("{"):
        content = '{"order": [' + content + ']}'
    order = RankOrder.model_validate_json(content).order
    print(order)
    return [chunks[i - 1] for i in order]

def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

# DEFINING FUNCTION THAT FETCH UNRANKED CHUNKS:
def fetch_context_unranked(question):
    # 1 Generate embedding for the question
    query_embedding = hf_embeddings.embed_query(question)  # your HuggingFace embedder
    # 2️ Query the Chroma collection
    results = collection.query(query_embeddings=[query_embedding], n_results=RETRIEVAL_K)
    # 3️ Convert results into Result objects
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks

# MAKING FUNCTION THAT FETCHES CONTEXT AND RERANKS IT
def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]

# DEFINING SYSTEM PROMPT FOR THE ANSWER_QUESTION FUNCTION:
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

# MAKE RAG MESSAGES:
def make_rag_messages(question, history, chunks):
    context = "\n\n".join(f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

# MAKING FUNCTION TO REWRITE QUERY:
#@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = openrouter.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "system", "content": message}],
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# @retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    query = rewrite_query(question, history)
    print(query)
    chunks = fetch_context(query)
    messages = make_rag_messages(question, history, chunks)
    response = openrouter.chat.completions.create(
        model="gpt-oss-120b",
        messages=messages,
        temperature=0,
        max_tokens=1000  
    )
    answer = response.choices[0].message.content.strip()
    return answer, chunks