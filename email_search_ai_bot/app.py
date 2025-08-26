import streamlit as st
from src.embedding_layer import EmbeddingProcessor
from src.search_layer import SearchEngine
from src.generation_layer import generate_answer

import chromadb
from chromadb.config import Settings

# Page config
st.set_page_config(
    page_title="📧 Email Search AI",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background styling
def add_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1549921296-3a3b3b2d5f1b");
            background-size: cover;
            background-attachment: fixed;
            color: blue;
        }}
        .stTextInput > div > div > input {{
            background-color: #ffffff33;
            color: blue;
        }}
        .stTextInput > label {{
            color: black;
        }}
        .stButton > button {{
            background-color: #262730;
            color: yellow;
            border-radius: 10px;
        }}
        .stButton > button:hover {{
            background-color: #4e4e57;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# Initialize Chroma client and models
@st.cache_resource(show_spinner=False)
def init_system():
    chroma_client = chromadb.Client(Settings(persist_directory="chroma_db"))
    embedder = EmbeddingProcessor(client=chroma_client)
    search_engine = SearchEngine(client=chroma_client)
    return embedder, search_engine

embedder, search_engine = init_system()

# Header
st.title("📧 Email Search AI Assistant")
st.markdown("Ask natural language questions across organizational email threads. 🧠")

# Sidebar for filtering
with st.sidebar:
    st.header("🔎 Query Settings")
    st.markdown("Use these options to narrow down your search.")

    top_k = st.slider("Number of top results (K)", min_value=1, max_value=10, value=3)

    filter_by_thread = st.checkbox("Filter by Thread ID")
    thread_id = None
    if filter_by_thread:
        thread_id = st.number_input("Enter Thread ID", min_value=1, step=1)

# Query box
#query = st.text_input("💬 Enter your query", placeholder="e.g., What strategy was proposed in thread_id 1001?")
# Query input and buttons
st.markdown("## 🔍 Ask a Question")
query_placeholder = st.empty()
query_text = query_placeholder.text_input("Enter your query", placeholder="e.g., What strategy was proposed in thread_id 1001?", key="query_input")

col1, col2 = st.columns([1, 1])

run_search = False

with col1:
    if st.button("🔍 Search"):
        if query_text.strip():
            run_search = True
        else:
            st.warning("Please enter a query before searching.")

with col2:
    if st.button("🔄 Clear"):
        st.session_state.query_input = ""
        st.experimental_rerun()


# Run Search
if run_search:
    with st.spinner("🔍 Searching relevant chunks..."):
        top_chunks = search_engine.search(query_text, top_k=top_k, filter_thread_id=thread_id)

    if not top_chunks:
        st.warning("❌ No relevant results found. Try rephrasing your query or check your filters.")
    else:
        st.markdown("---")
        st.subheader("🔍 Top Retrieved Chunks")
        for i, chunk in enumerate(top_chunks):
            with st.expander(f"📄 Chunk {i+1}"):
                st.markdown(f"**Content:** {chunk['chunk']}")
                meta = chunk.get("metadata", {})
                st.caption(f"📧 From: {meta.get('from', 'N/A')}")
                st.caption(f"🧵 Subject: {meta.get('subject', 'N/A')}")
                st.caption(f"🕒 Timestamp: {meta.get('timestamp', 'N/A')}")
                st.caption(f"📌 Thread ID: {meta.get('thread_id', 'N/A')}")

        # Generate Answer Button
        if st.button("🧠 Generate Answer from LLM"):
            with st.spinner("⚙️ Generating final answer..."):
                answer = generate_answer(query_text, top_chunks)
            st.markdown("---")
            st.subheader("✅ Final Answer")
            st.success(answer)
