# For storing pdf content in qdrant vectorstore
import os
import logging
from typing import List, Optional
from PyPDF2 import PdfReader
from configs import config

from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# qdrant client
from qdrant_client import QdrantClient

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PDFQdrantIndexer:
    """
    Class to create and persist PDF embeddings into a Qdrant vector store.
    """

    def __init__(
        self,
        pdf_folder: str,
        qdrant_path: Optional[str] = "./qdrant_pdf_data",
        collection_name: str = "pdf_embeddings",
        embedding_model_name: str = config.embedding_model_name,
        chunk_size: int = 2048,
        chunk_overlap: int = 128,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.pdf_folder = pdf_folder
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key

        self.documents: List[Document] = []
        self.nodes = None
        self.client: Optional[QdrantClient] = None
        self.vector_store = None
        self.storage_context: Optional[StorageContext] = None

        # set embedding model into Settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        logger.info(f"Embedding model set to: {self.embedding_model_name}")

    def _list_pdf_files(self) -> List[str]:
        if not os.path.isdir(self.pdf_folder):
            raise FileNotFoundError(f"PDF folder not found: {self.pdf_folder}")
        files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith(".pdf")]
        logger.info(f"Found {len(files)} PDF files in {self.pdf_folder}")
        return files

    def load_documents(self) -> List[Document]:
        pdf_files = self._list_pdf_files()
        documents: List[Document] = []

        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_folder, pdf_file)
            try:
                reader = PdfReader(file_path)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                continue

            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text = "\n".join(text_parts).strip()
            if text:
                doc = Document(text=text, metadata={"source": pdf_file})
                documents.append(doc)
            else:
                logger.warning(f"No extractable text in {pdf_file}; skipping.")

        self.documents = documents
        logger.info(f"Loaded {len(self.documents)} documents with text.")
        return self.documents

    def split_documents(self):
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.nodes = splitter.get_nodes_from_documents(self.documents)
        logger.info(f"Split documents into {len(self.nodes)} nodes.")
        return self.nodes

    def setup_qdrant(self):
        if self.qdrant_path:
            self.client = QdrantClient(path=self.qdrant_path)
            logger.info(f"Created local QdrantClient with path: {self.qdrant_path}")
        elif self.qdrant_host:
            url = self.qdrant_host
            kwargs = {}
            if self.qdrant_api_key:
                kwargs["api_key"] = self.qdrant_api_key
            if self.qdrant_port:
                url = f"{url}:{self.qdrant_port}"
            self.client = QdrantClient(url=url, **kwargs)
            logger.info(f"Created remote QdrantClient at: {url}")
        else:
            self.client = QdrantClient(path=self.qdrant_path or "./qdrant_pdf_data")
            logger.info(f"Created QdrantClient with default path: {self.qdrant_path or './qdrant_pdf_data'}")

        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.client,
            embedding=Settings.embed_model,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        logger.info(f"Qdrant vector store configured for collection: {self.collection_name}")

    def build_and_persist_index(self, use_nodes: bool = False):
        if self.storage_context is None:
            raise ValueError("Storage context not initialized. Call setup_qdrant() first.")
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        index_source = self.nodes if use_nodes and self.nodes else self.documents

        index = VectorStoreIndex.from_documents(
            documents=index_source,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model,
        )

        self.storage_context.persist()
        logger.info("Index built and storage context persisted to Qdrant.")
        self.close()
        return index

    def run_full_pipeline(self, use_nodes: bool = False):
        self.load_documents()
        if use_nodes:
            self.split_documents()
        self.setup_qdrant()
        index = self.build_and_persist_index(use_nodes=use_nodes)
        return {"documents_indexed": len(self.documents), "nodes_created": len(self.nodes) if self.nodes else 0}

    def close(self):
        """Close Qdrant client if present to release local storage lock."""
        try:
            if getattr(self, "client", None) is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
                finally:
                    self.client = None
        except Exception:
            pass