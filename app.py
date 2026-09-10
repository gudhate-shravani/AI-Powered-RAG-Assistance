import streamlit as st

from rag_pipeline import (
    process_document,
    ask_question
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ RAG Settings")

    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    st.divider()

    st.markdown("### 🔍 Retrieval")

    top_k = st.slider(
        "Relevant chunks",
        min_value=1,
        max_value=8,
        value=4
    )

    st.divider()

    st.markdown("### 🧠 Technology")

    st.write("🐍 Python")
    st.write("📄 PyPDF")
    st.write("🔤 Sentence Transformers")
    st.write("🔎 FAISS")
    st.write("🤖 Groq LLM")
    st.write("🎨 Streamlit")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("🤖 RAG Assistant")

st.markdown(
    """
### Ask questions about your PDF

Upload a document and ask questions based
on its content.
"""
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📚 Processing your PDF..."
        ):

            try:

                chunks, index = process_document(
                    uploaded_file
                )

                st.session_state.chunks = chunks

                st.session_state.index = index

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.messages = []

            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )

                st.stop()


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.index is not None:

    st.success(
        f"✅ Document ready: "
        f"{st.session_state.document_name}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Text Chunks",
            len(st.session_state.chunks)
        )

    with col2:

        st.metric(
            "Indexed Vectors",
            st.session_state.index.ntotal
        )

else:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "to start chatting."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander(
                "📚 View Sources"
            ):

                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Source Chunk {i}**"
                    )

                    st.write(
                        source["chunk"]
                    )

                    st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask something about your PDF..."
)


# ============================================================
# ASK RAG BACKEND
# ============================================================

if query:

    if st.session_state.index is None:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching document..."
        ):

            try:

                answer, sources = ask_question(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k
                )

                st.markdown(answer)


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                with st.expander(
                    "📚 View Sources"
                ):

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"**Source Chunk {i}**"
                        )

                        st.write(
                            source["chunk"]
                        )

                        st.divider()


                # ------------------------------------------------
                # SAVE MESSAGE
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


            except Exception as e:

                error_message = (
                    f"Something went wrong: {e}"
                )

                st.error(
                    error_message
                )