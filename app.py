import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
from main import load_data, chunk_data, embed_and_store, search_query, generate_response

st.title("RAG Chat over Structured Data (Excel/CSV/JSON/PDF Supported)")


if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'df' not in st.session_state: 
    st.session_state.df = None

# File upload
uploaded_file = st.file_uploader("Upload your file", type=['xlsx', 'csv', 'json', 'pdf'])
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1]
    temp_path = f"temp_file.{file_ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    try:
        documents = load_data(temp_path)
        chunks = chunk_data(documents, file_path=temp_path)
        vectorstore = embed_and_store(chunks)
        st.session_state.vectorstore = vectorstore
        st.session_state.temp_path = temp_path
        st.session_state.history = [] 
      
        if file_ext in ['xlsx', 'csv']:
            st.session_state.df = pd.read_excel(temp_path) if file_ext == 'xlsx' else pd.read_csv(temp_path)
        st.success("File ingested! Ready for queries.")
    except ValueError as e:
        st.error(str(e))


# Chat interface
user_query = st.text_input("Ask a question :")
if user_query:
    if st.session_state.vectorstore is None:
        st.warning("Upload a file first.")
    else:
        try:
            retrieved = search_query(user_query, st.session_state.vectorstore)
            response = generate_response(user_query, retrieved, st.session_state.history)
            st.session_state.history.append(f"User: {user_query}\nAI: {response}")
            st.markdown("**Response:**")
            st.markdown(response)
            
           
            if any(word in user_query.lower() for word in ["chart", "graph", "plot", "visualize"]):
                if st.session_state.df is not None:
                    try:
                        # Parse query for columns 
                        words = user_query.lower().split()
                        x_col = next((w for w in words if w in st.session_state.df.columns), None)
                        y_col = next((w for w in words[::-1] if w in st.session_state.df.columns and w != x_col), None)
                        if x_col and y_col:
                            fig, ax = plt.subplots()
                            ax.plot(st.session_state.df[x_col], st.session_state.df[y_col], marker='o')
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            ax.set_title(f"{y_col} vs {x_col}")
                            st.pyplot(fig)
                        else:
                            st.write("Chart not generated: Specify columns like 'chart displacement vs draft'.")
                    except Exception as e:
                        st.write(f"Chart error: {str(e)}")
                else:
                    st.write("Charts supported only for Excel/CSV.")
        except Exception as e:
            st.error(f"Query error: {str(e)}")

# Display history
st.text_area("Conversation History", value="\n".join(st.session_state.history), height=200)



if st.button("Clear Session (Delete Temp File)"):
    if st.session_state.temp_path and os.path.exists(st.session_state.temp_path):
        os.remove(st.session_state.temp_path)
    st.session_state.clear()
    st.success("Session cleared.")
