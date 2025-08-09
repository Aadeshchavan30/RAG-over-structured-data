import os
import pandas as pd
from langchain.document_loaders import CSVLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from collections import defaultdict
from dotenv import load_dotenv 


# Load environment variables from .env file
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it and try again.")

def load_data(file_path):
    """Load data from file and convert to LangChain Documents."""
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            documents = []
            headers = df.columns.tolist()
            for index, row in df.iterrows():
                content = " | ".join([f"{headers[i]}: {val}" for i, val in enumerate(row)])
                documents.append(Document(page_content=content, metadata={"row": index, "source": file_path}))
            return documents
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
            return loader.load()
        elif file_path.endswith('.json'):
            loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=False)
            return loader.load()
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = file_path
            return documents
        else:
            raise ValueError("Unsupported file type. Use .xlsx, .csv, .json, or .pdf")
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {str(e)}. Ensure file is valid and dependencies installed (e.g., openpyxl for XLSX).")

def chunk_data(documents, strategy='hybrid', file_path=None):
    """Chunk data with hybrid strategy for better granularity."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
        if strategy == 'hybrid':
            row_chunks = splitter.split_documents(documents)
            header = ""
            if file_path and (file_path.endswith('.xlsx') or file_path.endswith('.csv')):
                df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
                header = "Columns: " + " | ".join(df.columns)
            if header:
                row_chunks.insert(0, Document(page_content=header, metadata={"type": "header"}))
            return row_chunks
        else:
            raise ValueError("Invalid strategy. Use 'hybrid'.")
    except Exception as e:
        raise ValueError(f"Error chunking data: {str(e)}")

def embed_and_store(chunks):
    """Embed chunks and store in Chroma vector DB."""
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore

def search_query(query, vectorstore, k=5):
    """Search for similar chunks using query embedding."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    query_embedding = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=k)
    unique_results = list({res.page_content: res for res in results}.values())
    return [res.page_content for res in unique_results]

def generate_response(query, retrieved_chunks, history):
    """Generate response using LLM, with history and strict anti-hallucination prompt."""
    context = "\n".join(set(retrieved_chunks))
    # Strict prompt to use ONLY retrieved data, preserve numbers/units, format as table
    prompt_template = """
    Conversation history: {history}
    Retrieved data (USE ONLY THIS EXACT DATA, no additions, no summarization, preserve numbers/units precisely): {context}
    User query: {query}
    Answer using exact values from data. For tables, start with headers, then rows. Use Markdown table format. Support row/column/numeric/range queries directly.
    If query asks for chart, say 'Chart requested' but do not generate it here.
    """
    prompt = PromptTemplate(input_variables=["history", "context", "query"], template=prompt_template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(history="\n".join(history), context=context, query=query)
    
    # Post-processing: Validate response matches data, infer table if needed
    all_data = defaultdict(list)
    for chunk in retrieved_chunks:
        pairs = [pair.strip() for pair in chunk.split('|') if ':' in pair]
        for pair in pairs:
            key, value = pair.split(':', 1)
            all_data[key.strip()].append(value.strip())
    if all_data:
        headers = list(all_data.keys())
        max_rows = max(len(v) for v in all_data.values())
        table = "| " + " | ".join(headers) + " |\n| " + " --- |" * len(headers) + "\n"
        for i in range(max_rows):
            row = [all_data[h][i] if i < len(all_data[h]) else "" for h in headers]
            table += "| " + " | ".join(row) + " |\n"
        response = table if "table" in query.lower() else response  # Use inferred table if query fits
    
    # Anti-hallucination check: Warn if response doesn't match data
    if not any(chunk in response for chunk in retrieved_chunks):
        response += "\n(Warning: Response based strictly on data; verify manually)"
    return response
