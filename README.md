# Information Retrieval (IR) System with RAG using Qdrant & Streamlit   

## **Task Objective**:   
Develop a domain-specific Information Retrieval (IR) system with the following key components:   
**1. Data Layer**: A large, domain-focused corpus stored as vector embeddings within a vector database.   
**2. Query Intelligence**: A query suggestion module that enhances user input for better retrieval accuracy.   
**3. Ranking Engine**: Capability to return top-ranked responses with associated relevance metrics.   
**4. User Interface**: An intuitive front-end interface enabling user interaction and real-time query processing.   
**5.Implementation Language**: Python.


This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying domain-specific PDFs.  
It combines **semantic search (Qdrant vector database)** and **LLM-based response generation**, all through a **Streamlit UI**.

## Features

- **PDF Ingestion & Indexing**
  - Upload or select local folders of PDFs
  - Extract text, chunk it intelligently
  - Generate embeddings using Hugging Face models
  - Store embeddings in **Qdrant Vector Store**

- **Retrieval-Augmented Generation (RAG)**
  - Query stored PDFs in natural language
  - Retrieve relevant text chunks from Qdrant
  - Rerank results using LLM
  - Generate concise answers using OpenAI-compatible LLMs

- **Interactive Streamlit Interface**
  - Upload or select PDFs for indexing
  - Ask queries directly from UI
  - Explore retrieved documents with relevance scores

## Project Structure
IR_System_with_RAG/
│   
├── configs/   
│ └── config.py # Model names and configuration   
│   
├── helper_for_indexing.py # Handles PDF parsing, embedding, and Qdrant indexing   
├── helper_for_query_system.py # Handles query, retrieval, and LLM response generation    
├── app.py # Streamlit UI (main entry point)    
│    
├── .env # Environment variables (API keys)    
├── requirements.txt # Dependencies list   
   



