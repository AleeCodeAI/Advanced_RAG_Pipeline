# ====================================== IMPORTING LIBRARIES =========================================
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
import os
from langchain_huggingface import HuggingFaceEmbeddings
import json
import re
from tenacity import retry, wait_exponential
from litellm import completion
from multiprocessing import Pool

# ==================================== CREDENTIALS =========================================
load_dotenv(override=True)

# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
# openrouter_url = os.getenv("OPENROUTER_URL")
# 
# openrouter = OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)
# 
MODEL = "openrouter/openai/gpt-oss-120b"

DB_NAME = "preprocessed_db"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 600

collection_name = "docs"
embedding_model = "all-MiniLM-L6-v2"
WORKERS = 4
wait = wait_exponential(multiplier=1, min=10, max=240)
# ========================================================================================
                                    # STEP 1: CHUNKING
# ========================================================================================

# DEFINING THE DATA TEMPLATES:
class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text, metadata=metadata)
    
class Chunks(BaseModel):
    chunks: list[Chunk]

# FETCHING DOCUMENTS:
def fetch_documents():
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents

# MAKING A FUNCTION FOR LLM FOR CHUNKING:

def make_prompt(document, average_chunk_size=1000):
    how_many = (len(document["text"]) // average_chunk_size) + 1
    
    # Get the schema
    schema = Chunks.model_json_schema()
    
    return f"""
You take a document and you split it into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words.

For each chunk, provide a headline, summary, and the original text of the chunk.

IMPORTANT: Respond ONLY with valid JSON. Do not include any markdown formatting, code blocks, or explanatory text.

Use this exact JSON schema:
{json.dumps(schema, indent=2)}

Here is the document:

{document["text"]}
"""

# DEFINING MESSAGES TO PASS TO LLM:
def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]

@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

# FINALLY MAKE THE FUNCTION FOR CHUNKING
def create_chunks(documents):
    """
Create chunks using a number of workers in parallel.
If you get a rate limit error, set the WORKERS to 1.
"""
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len
    (documents)):
            chunks.extend(result)
    return chunks


# ========================================================================================
                                    # STEP 2: EMBEDDING
# ========================================================================================

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    # Delete existing collection if exists
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    # Extract texts
    texts = [chunk.page_content for chunk in chunks]
    # Generate embeddings using LangChain HuggingFace wrapper
    vectors = hf_embeddings.embed_documents(texts)
    # Create new collection
    collection = chroma.get_or_create_collection(collection_name)
    # Prepare IDs and metadata
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    # Add vectors to Chroma
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


# =====================================================================================
                                # INGESTING DATA
# ====================================================================================

if __name__ == "__main__":
    # FETCHING AND CHUNKING DATA:
    documents = fetch_documents()
    print("documents.fetched")
    chunks = create_chunks(documents)
    print(f"{len(chunks)} Chunks made!")
    # CREATE EMBEDDINGS AND VECTORSTORE
    create_embeddings(chunks)
    print("INGESTION COMPLETED!!!")




# def extract_json(text: str) -> str:
    # """
    # Extract the largest JSON object from the LLM output.
    # """
    # match = re.search(r"\{.*\}", text, re.DOTALL)
    # if not match:
        # raise ValueError("No JSON object found in LLM output")
    # return match.group(0)
# 

# def process_document(document, max_retries=3):
    # messages = make_messages(document)
    # for attempt in range(max_retries):
        # try:
            # response = openrouter.chat.completions.create(
                # model="gpt-3.5-turbo",
                # messages=messages,
                # temperature=0,
                # max_tokens=4000,
                # response_format={"type": "json_object"}
            # )
            # content = response.choices[0].message.content.strip()

            # try:
                # doc_as_chunks = Chunks.model_validate_json(content).chunks
            # except:

                # json_content = extract_json(content)
                # doc_as_chunks = Chunks.model_validate_json(json_content).chunks
            # return [chunk.as_result(document) for chunk in doc_as_chunks]
        # except Exception as e:
            # if attempt == max_retries - 1:
                # print(f"Failed after {max_retries} attempts for: {document['source']}")
                # raise
            # print(f"Attempt {attempt + 1} failed, retrying...")
            # continue
