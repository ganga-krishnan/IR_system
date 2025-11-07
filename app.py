# For UI
import os
import tempfile
import shutil
import streamlit as st
import tkinter as tk
from configs import config
from tkinter import filedialog
from dotenv import load_dotenv
load_dotenv("/home/ganga/Desktop/inerg_task/IR_System_with_RAG/.env")

try:
    from helper_for_indexing import PDFQdrantIndexer
    INDEXER_MODULE = 'indexer_app'
except Exception:
    PDFQdrantIndexer = None
    INDEXER_MODULE = None

try:
    from helper_for_query_system import QdrantRAGQuery
    RAG_MODULE = 'retrieval_app'
except Exception:
    QdrantRAGQuery = None
    RAG_MODULE = None

# Enable wide layout
st.set_page_config(page_title="PDF RAG System", layout="wide")
st.title("Information Retrieval System")

# --- Default configuration (no sidebar) ---
qdrant_path = "./qdrant_pdf_data"
collection_name = "pdf_embeddings"
embedding_model_name = config.embedding_model_name
llm_model_name = config.llm_model_name
openai_api_key = os.getenv("OPENAI_API_KEY")  

# Optional indexing
with st.form(key="index_form"):
    st.subheader("Index New Documents into database")

    use_local_folder = st.checkbox("Use local folder path (server-accessible)", value=True)
    uploaded_files = None
    local_folder = ""

    if use_local_folder:
        col1, col2 = st.columns([1, 3])
        with col1:
            choose_folder = st.form_submit_button("Choose Folder")
        with col2:
            local_folder = st.session_state.get("selected_folder", "")
            st.text_input("Selected folder path:", value=local_folder, disabled=True)

        # If user clicks the "Choose Folder" button
        if choose_folder:
            root = tk.Tk()
            root.withdraw()
            selected_folder = filedialog.askdirectory(title="Select PDF Folder")
            root.destroy()
            if selected_folder:
                st.session_state["selected_folder"] = selected_folder
                local_folder = selected_folder
                st.success(f"Selected folder: {selected_folder}")

    else:
        uploaded_files = st.file_uploader("Upload PDF files (multiple)", type=["pdf"], accept_multiple_files=True)

    # Main action button — clear and separate
    st.markdown("---")
    submit_index = st.form_submit_button(label="Upload & Index PDFs")

    if submit_index:
        if PDFQdrantIndexer is None:
            st.error("PDFQdrantIndexer class not found. Make sure 'helper_for_indexing.py' exists in the app folder.")
        else:
            try:
                # Prepare folder to index
                if use_local_folder:
                    local_folder = st.session_state.get("selected_folder", "")
                    if not local_folder or not os.path.isdir(local_folder):
                        st.error(f"Local folder not found: {local_folder}")
                        index_folder = None
                    else:
                        index_folder = local_folder
                else:
                    if not uploaded_files:
                        st.error("No files uploaded")
                        index_folder = None
                    else:
                        tmp = tempfile.mkdtemp(prefix="streamlit_uploaded_pdfs_")
                        for f in uploaded_files:
                            out_path = os.path.join(tmp, f.name)
                            with open(out_path, "wb") as out_file:
                                out_file.write(f.getbuffer())
                        index_folder = tmp
                        st.info(f"Saved {len(uploaded_files)} files to temporary folder: {tmp}")

                # Run indexing if folder found
                if index_folder:
                    st.info("Starting indexing — may take time depending on PDF size.")
                    with st.spinner("Indexing PDFs into Qdrant..."):
                        indexer = PDFQdrantIndexer(
                            pdf_folder=index_folder,
                            qdrant_path=qdrant_path,
                            collection_name=collection_name,
                            embedding_model_name=embedding_model_name,
                        )
                        result = indexer.run_full_pipeline(use_nodes=False)

                    st.success("Indexing completed successfully!")
                    st.json(result)

                # Cleanup temp directory
                if not use_local_folder and 'tmp' in locals() and os.path.isdir(tmp):
                    try:
                        shutil.rmtree(tmp)
                    except Exception:
                        pass

            except Exception as e:
                st.exception(e)


# --- Minimal Query UI (only question + run) ---
st.header("Ask a Question")
st.write("Enter your question below and click **Run query**. The request runs synchronously.")

with st.form(key="query_form"):
    user_query = st.text_area("Your question", height=140)
    submit_query = st.form_submit_button(label="Run query")

    if submit_query:
        if not user_query or user_query.strip() == "":
            st.error("Please enter a question to query.")
        elif QdrantRAGQuery is None:
            st.error(
                "QdrantRAGQuery class not found. Make sure 'retrieval_app.py' exists in the app folder."
            )
        else:
            rag = None
            try:
                with st.spinner("Running RAG query — this may take a few seconds..."):
                    rag = QdrantRAGQuery(
                        qdrant_path=qdrant_path,
                        collection_name=collection_name,
                        embedding_model_name=embedding_model_name,
                        llm_model_name=llm_model_name,
                        openai_api_key=openai_api_key or None,
                    )

                    rag.load_vector_store()
                    rag.attach_llm()
                    rag.create_query_engine(similarity_top_k=5)  # fixed default; change in code if needed

                    full_output = rag.query(user_query, top_k=5)

                    answer = full_output.get("Response")

                    # store full output in session state for later exploration
                    st.session_state["last_final_output"] = full_output

                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.exception(e)

if st.session_state.get("last_final_output") is not None:
    if st.button("Explore more"):
        full_output = st.session_state["last_final_output"]
        with st.expander("Explore more — full output and retrieved documents", expanded=True):
            st.write("Full RAG output:")
            st.json(full_output)

else:
    st.info("Run a query first to enable 'Explore more'.")