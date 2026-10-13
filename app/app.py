import os
import sys
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_PATH)


# ============================================================
# IMPORT RAG FUNCTIONS
# ============================================================

import rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BPO Knowledge Copilot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 BPO Knowledge Copilot")

st.write(
    "Ask a customer-support question and get a "
    "knowledge-grounded recommendation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This Copilot uses Retrieval-Augmented Generation (RAG)
        to retrieve relevant BPO knowledge and generate a
        grounded response for support agents.
        """
    )

    st.divider()

    st.subheader("Retrieval")

    st.write("Hybrid Retrieval")

    st.write(
        "• 60% Semantic Embeddings\n"
        "• 40% TF-IDF"
    )

    st.divider()

    st.caption(
        "BPO Knowledge Copilot"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask your question")

question = st.text_area(
    "Customer / Agent Question",
    placeholder=(
        "Example: The customer says their claim was "
        "rejected even though they had coverage. "
        "What should I do?"
    ),
    height=120
)


# ============================================================
# ASK COPILOT BUTTON
# ============================================================

if st.button(
    "🔍 Ask Copilot",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        # ----------------------------------------------------
        # Retrieve knowledge
        # ----------------------------------------------------

        with st.spinner(
            "Searching the knowledge base..."
        ):

            results = rag.hybrid_retrieve(
                question,
                top_k=3
            )

            context = rag.build_context(
                results
            )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = rag.generate_answer(
                question,
                context
            )


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.subheader(
            "💡 Copilot Recommendation"
        )

        st.write(answer)


        # ----------------------------------------------------
        # Display retrieved sources
        # ----------------------------------------------------

        st.subheader(
            "📚 Retrieved Knowledge"
        )

        for _, row in results.iterrows():

            with st.expander(
                f"{row['article_id']} — {row['title']}"
            ):

                st.write(
                    f"*Hybrid Score:* "
                    f"{row['hybrid_score']:.4f}"
                )

                page = row.get(
                    "page",
                    row.get(
                        "page_number",
                        "Unknown"
                    )
                )

                st.write(
                    f"*Page:* {page}"
                )

                st.write(
                    row.get(
                        "text",
                        ""
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Hybrid Retrieval + RAG + OpenRouter LLM"
)