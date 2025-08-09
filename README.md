# RAG Over Structured Data

This repository implements a Retrieval-Augmented Generation (RAG) system for querying structured datasets (e.g., Excel, CSV, JSON, PDF) via natural language. It uses LangChain for the pipeline, Google Gemini for embeddings and LLM, Chroma as the vector database, and Streamlit for a simple UI. The system preserves data granularity without summarization, supports row/column/numeric/range queries, and includes bonus chart visualization.

The overall architecture follows a modular, robust RAG design optimized for structured data:
<img width="1080" height="677" alt="image" src="https://github.com/user-attachments/assets/6174bb6c-d038-4d1d-90e7-2c371e876b75" />

# RAG System Architecture Description

1. Data Ingestion: User uploads a structured file (Excel, CSV, JSON, or PDF).

2. Chunking: The data is split into granular “chunks” (usually row-wise or hybrid, ensuring no loss of detail or numeric precision).

3. Embedding: Each chunk is passed through the Gemini embedding model to create vector representations.

4. Vector Database (Chroma): Chunk embeddings are stored in a local vector database for efficient similarity search and retrieval.

5. Query Embedding: The user’s natural language question is embedded using Gemini to the same vector space.

6. Semantic Search: The system retrieves the top-K most relevant data chunks from the vector database matching the query embedding.

7. Context Construction: The results (retrieved chunks) and user query are combined into a system prompt/context.

8. LLM Generation: The prompt/context is sent to the Gemini LLM to answer the user’s question, strictly referencing only the retrieved data (no hallucination/guessing).

9. Conversation History: Every question/answer is stored in session state to allow for natural, contextual follow-up conversation.


## Data Ingestion Approach
Data ingestion starts with loading files into LangChain Documents to preserve structure and metadata:

* **Supported Formats**: Excel (.xlsx) via Pandas (converts rows to key-value strings like "Draft: 5 | Displacement: 1000"), CSV/JSON via LangChain loaders, PDF via PyPDFLoader (extracts text with source metadata).

* **Process**: The load_data function reads the file, extracts headers/rows, and creates Documents with metadata (e.g., row index). This ensures no data loss—exact values and context are retained.

* **Why This Approach?** It handles tabular data granularly (row-by-row) without flattening or summarizing, aligning with the assignment's need for precision over structured datasets. For Excel/CSV, Pandas integration allows reliable header extraction and numeric preservation.


## Retrieval Strategies and Why I Chose Them
Retrieval uses semantic search on embedded chunks to fetch relevant data without hallucinations.

* **Chunking Strategy**: Hybrid (row-wise splitting with RecursiveCharacterTextSplitter + header injection). Chunks are small (size 150, overlap 50) for precision. Headers are prepended as a special Document for context.

Why Hybrid? Row-wise preserves table structure for queries like "row 3 details," while headers enable column-based filtering. I experimented with cell-based and chose hybrid for best balance of granularity and recall

* **Embedding and Storage:** Gemini's "embedding-001" model for vectorization, stored in Chroma. Top-K=5 similarity search with deduplication.

Why Chroma and This Setup? Chroma is lightweight and fast for local dev (vs. Pinecone for cloud scalability—I considered it but stuck with local for simplicity). Semantic search ensures relevant chunks are retrieved; deduplication avoids redundancy. This optimizes for assignment criteria like retrieval accuracy and speed.

* **Performance Comparison:** Hybrid chunking outperformed row-only (better on range queries) and cell-based (fewer hallucinations).


## Pre/Post-Processing Optimizations Implemented
To maximize accuracy and avoid hallucinations:

* **Pre-Processing:** During ingestion/chunking, numeric precision is preserved (no rounding via string formatting). Headers are explicitly added to chunks for context. Query embeddings use the same Gemini model for consistency.

* **Post-Processing:** In generate_response, retrieved chunks are deduplicated and fed to Gemini with a strict prompt ("USE ONLY THIS EXACT DATA, no additions"). Response is validated (warn if it doesn't match chunks). For tables, we infer headers/rows and format as Markdown (e.g., auto-builds "| Draft | Displacement |" from key-values).

* **Optimizations for Quality**: Lower LLM temperature (0.1) for factual responses. History integration supports follow-ups. These reduce hallucinations (e.g., prompt forbids summarization) and ensure unit consistency (e.g., "tons" preserved).

## Limitations and Future Improvements
* **Limitations:** PDF handling is text-only (no native table extraction, may lose structure without OCR). Charts are basic (Matplotlib line plots; assumes column names in query). Local Chroma can lock files on Windows, causing deletion issues during uploads. No multi-modal support yet (e.g., image tables). Free Gemini API has rate limits.

* **Future Improvements:** Add OCR (e.g., EasyOCR) for image/PDF tables (bonus multi-modal). Switch to Pinecone for cloud storage (avoids local locks, scales better). Implement advanced chunking (e.g., semantic-based with LLMs) and cost tracking (e.g., token usage). Enhance UI with React for better UX. Add evaluation metrics (e.g., automated accuracy tests on query types).
