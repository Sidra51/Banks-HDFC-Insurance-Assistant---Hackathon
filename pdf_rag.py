import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_openai import ChatOpenAI
import httpx
from langchain.llms.base import LLM
from typing import List, Any, Dict, Optional, Tuple, Mapping
from pydantic import PrivateAttr
from groq import Groq
from langchain_community.vectorstores import Chroma
import chromadb
from rank_bm25 import BM25Okapi
from langchain.schema.retriever import BaseRetriever

# Load environment variables
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
DB_PATH = "chromadb"
PENSION_COLLECTION = "pension_plans"
ULIP_COLLECTION = "ulip_plans"
PROTECTION_COLLECTION = "protection_plans"
HEALTH_COLLECTION = "health_plans"
SAVINGS_COLLECTION = "savings_plans"
ANNUITY_COLLECTION = "annuity_plans"
ALL_POLICIES_COLLECTION = "all_policies"

client = chromadb.PersistentClient(path=DB_PATH)
pension_collection = client.get_or_create_collection(name=PENSION_COLLECTION)
ulip_collection = client.get_or_create_collection(name=ULIP_COLLECTION)
protection_collection = client.get_or_create_collection(name=PROTECTION_COLLECTION)
health_collection = client.get_or_create_collection(name=HEALTH_COLLECTION)
savings_collection = client.get_or_create_collection(name=SAVINGS_COLLECTION)
annuity_collection = client.get_or_create_collection(name=ANNUITY_COLLECTION)
all_policies_collection = client.get_or_create_collection(name=ALL_POLICIES_COLLECTION)

COLLECTION_MAP = {
    "pension_plans": pension_collection,
    "ulip_plans": ulip_collection,
    "protection_plans": protection_collection,
    "health_plans": health_collection,
    "savings_plans": savings_collection,
    "annuity_plans": annuity_collection,
    "all_policies": all_policies_collection
}

# Initialize device and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

# Custom Vector Store
class SentenceTransformerVectorStore(VectorStore):
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self._documents = []
        self._embeddings = []
        self._metadatas = []

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        new_embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        self._documents.extend(documents)
        self._embeddings.extend(new_embeddings)
        self._metadatas.extend(metadatas)
        return [str(i) for i in range(len(self._documents))]

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        documents = [Document(page_content=text, metadata=meta or {}) 
                     for text, meta in zip(texts, metadatas or [{}] * len(texts))]
        return self.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        if not self._embeddings:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self._documents[i] for i in top_k_indices]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[Tuple[Document, float]]:
        if not self._embeddings:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(self._documents[i], similarities[i]) for i in top_k_indices]

    @classmethod
    def from_texts(cls, texts: List[str], embedding_model: SentenceTransformer, metadatas: Optional[List[dict]] = None, **kwargs):
        vector_store = cls(embedding_model)
        vector_store.add_texts(texts, metadatas)
        return vector_store

    @classmethod
    def from_documents(cls, documents: List[Document], embedding_model: SentenceTransformer, **kwargs):
        vector_store = cls(embedding_model)
        vector_store.add_documents(documents)
        return vector_store

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> Optional[bool]:
        if ids is None:
            return False
        indices_to_delete = [int(id_) for id_ in ids if id_.isdigit() and int(id_) < len(self._documents)]
        for index in sorted(indices_to_delete, reverse=True):
            if 0 <= index < len(self._documents):
                del self._documents[index]
                del self._embeddings[index]
                del self._metadatas[index]
        return True

# Custom Groq LLM Wrapper
class GroqLLMWrapper(LLM):
    model: str
    temperature: float = 0.0
    max_tokens: int = 1000
    _groq_client: Any = PrivateAttr()

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000, **kwargs):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._groq_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=30,
            stream=False
        )
        return response.choices[0].message.content

    async def _stream(self, prompt: str, stop: Optional[List[str]] = None):
        messages = [{"role": "user", "content": prompt}]
        stream = self._groq_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens}

# Custom Hybrid Retriever
class HDFCHybridRetriever(BaseRetriever):
    def __init__(self, collections: List[Any], embedding_model: Any, reranker: Any):
        super().__init__()
        self._collections = collections
        self._embedding_model = embedding_model
        self._reranker = reranker
        self._bm25_indexes = {}
        for collection in self._collections:
            try:
                stored_data = collection.get()
                stored_docs = stored_data['documents'] if 'documents' in stored_data else []
                tokenized_corpus = [doc.split() for doc in stored_docs]
                self._bm25_indexes[collection.name] = {
                    'index': BM25Okapi(tokenized_corpus),
                    'docs': stored_docs
                }
            except Exception as e:
                print(f"Error initializing BM25 for {collection.name}: {e}")
                self._bm25_indexes[collection.name] = {'index': None, 'docs': []}

    def _hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        combined_docs = []
        query_embedding = self._embedding_model.encode(query)
        for collection in self._collections:
            try:
                vector_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                if "documents" in vector_results:
                    for res in vector_results["documents"]:
                        combined_docs.extend(res)
            except Exception as e:
                print(f"Error in vector search for {collection.name}: {e}")
            try:
                bm25_data = self._bm25_indexes[collection.name]
                if bm25_data['index'] is not None:
                    tokenized_query = query.split()
                    bm25_scores = bm25_data['index'].get_scores(tokenized_query)
                    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
                    combined_docs.extend([bm25_data['docs'][i] for i in top_indices])
            except Exception as e:
                print(f"Error in BM25 search for {collection.name}: {e}")
        combined_docs = list(dict.fromkeys(combined_docs))
        if combined_docs:
            try:
                query_doc_pairs = [(query, doc) for doc in combined_docs]
                scores = self._reranker.predict(query_doc_pairs)
                combined_docs = [doc for _, doc in sorted(zip(scores, combined_docs), reverse=True)]
            except Exception as e:
                print(f"Error in reranking: {e}")
        return combined_docs[:top_k]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self._hybrid_search(query)
        return [Document(page_content=doc) for doc in docs]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# Initialize RAG components
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)

def get_llm(llm_choice: str):
    if llm_choice == "openrouter":
        return ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            api_key=os.environ.get("LLM_API_KEY"),
            model_name="meta-llama/llama-3.1-8b-instruct:free",
            streaming=True,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        )
    else:
        return GroqLLMWrapper(model="llama-3.1-8b-instant", temperature=0)

def load_knowledge_base(pdf_path, llm_choice):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_chunks = text_splitter.split_documents(documents)
    knowledge_base = SentenceTransformerVectorStore.from_documents(text_chunks, embedding_model)
    llm = get_llm(llm_choice)
    return RetrievalQA.from_chain_type(llm, retriever=knowledge_base.as_retriever())
