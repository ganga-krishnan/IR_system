
import os
import logging
from typing import Optional
from dotenv import load_dotenv
load_dotenv("/home/ganga/Desktop/inerg_task/IR_System_with_RAG/.env")

from configs import config

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------
# QdrantRAGQuery Class
# --------------------------------------------------
class QdrantRAGQuery:
    """
    A lightweight wrapper around a Qdrant VectorStore + LlamaIndex Query Engine.
    Used for querying previously stored document embeddings.
    """

    def __init__(
        self,
        qdrant_path: str = "./qdrant_pdf_data",
        collection_name: str = "pdf_embeddings",
        embedding_model_name: str = config.embedding_model_name,
        llm_model_name: str = config.llm_model_name,
        openai_api_key: Optional[str] = None,
    ):
        """
        :param qdrant_path: Local Qdrant storage path (must match the one used during embedding)
        :param collection_name: Name of the Qdrant collection to connect to
        :param embedding_model_name: Hugging Face embedding model name
        :param llm_model_name: OpenAI (or compatible) LLM model name
        :param openai_api_key: OpenAI API key (optional if already set in env)
        """
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Set as env var or pass explicitly.")

        # initialize embedding and LLM
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None

    # --------------------------------------------------
    # Core setup functions
    # --------------------------------------------------
    def load_vector_store(self):
        """Connect to local Qdrant and load the stored index."""
        logger.info(f"Connecting to local Qdrant at path: {self.qdrant_path}")

        # set embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)

        # connect Qdrant
        client = QdrantClient(path=self.qdrant_path)

        vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=client,
            embedding=Settings.embed_model,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("Loading VectorStoreIndex from persisted Qdrant data...")
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        logger.info("Vector index loaded successfully.")
        return self.index

    def attach_llm(self):
        """Attach LLM to the Settings (default OpenAI small model)."""
        logger.info(f"Attaching LLM model: {self.llm_model_name}")

        Settings.llm = OpenAI(
            model=self.llm_model_name,
            temperature=0.3,
            max_tokens=512
        )

        logger.info("LLM successfully attached.")
        return Settings.llm

    def create_query_engine(
        self,
        similarity_top_k: int = 20,
        rerank_top_n: int = 5,
        choice_batch_size: int = 10,
    ):
        """
        Create a RetrieverQueryEngine that:
        1) retrieves `similarity_top_k` candidate nodes from the vector index
        2) reranks them with an LLM reranker and returns top `rerank_top_n`
        """
        if not self.index:
            raise ValueError("Index not loaded. Call load_vector_store() first.")
        if not Settings.llm:
            self.attach_llm()

        # Retriever: retrieve more candidates before reranking
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
        )

        # Reranker: use the attached LLM to rerank retrieved nodes
        reranker = LLMRerank(
            llm=Settings.llm,
            top_n=rerank_top_n,
            choice_batch_size=choice_batch_size,
        )

        # Query engine: wire retriever + reranker together
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker],  # apply reranking as a post-processor
        )

        logger.info(
            f"RetrieverQueryEngine ready (retrieve={similarity_top_k}, rerank_top_n={rerank_top_n})"
        )
        return self.query_engine

    # --------------------------------------------------
    # Query function
    # --------------------------------------------------
    def query(self, user_query: str, top_k: int = 5) -> str:
        """Run a RAG-style query."""
        if not self.query_engine:
            self.create_query_engine(similarity_top_k=top_k)

        logger.info(f"Processing query: {user_query}")

        retriever = self.query_engine.retriever

        # Retrieve initial results from vector store
        retrieved_nodes = retriever.retrieve(user_query)

        # Apply reranking
        reranked_nodes = []
        for processor in self.query_engine._node_postprocessors:
            retrieved_nodes = processor.postprocess_nodes(retrieved_nodes, query_str=user_query)
            reranked_nodes = retrieved_nodes  # after reranking
            
        response = self.query_engine.query(user_query)
        # Build relevance metrics response
        results = []
        for i, node in enumerate(reranked_nodes[:top_k]):
            results.append({
                "rank": i + 1,
                "text_preview": node.text[:200] + "...",
                "retrieval_score": round(float(node.score), 4) if node.score else None,
            })

        final_output = {
            "Response": str(response),
            "top_ranked_documents": results
        }

        logger.info("Query complete with relevance metrics.")
        return final_output