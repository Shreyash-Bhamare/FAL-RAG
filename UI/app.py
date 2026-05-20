# ui/app.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.rag_pipeline import FALRAgPipeline

st.set_page_config(page_title="FAL-RAG", page_icon="⚖️", layout="wide")

st.title("⚖️ Faithfulness-Aware Legal RAG")
st.markdown("Automatic Detection and Mitigation of Contextual Hallucinations in the Legal Domain")

if "pipeline" not in st.session_state:
    with st.spinner("Initializing pipeline..."):
        st.session_state.pipeline = FALRAgPipeline()

pipeline = st.session_state.pipeline

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Enter your legal question:", placeholder="E.g., What is Article 21?")

with col2:
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)

if query:
    with st.spinner("Processing..."):
        result = pipeline.process_query(query, top_k=top_k)

    st.markdown("---")

    st.subheader("📄 Generated Answer")
    st.write(result["answer"])

    st.markdown("---")

    st.subheader("🔍 Hallucination Analysis")

    total_spans = len(result["span_analysis"])
    supported_count = sum(1 for s in result["span_analysis"] if s["status"] == "supported")
    uncertain_count = sum(1 for s in result["span_analysis"] if s["status"] == "uncertain")
    hallucinated_count = len(result["hallucinated_spans"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spans", total_spans)
    with col2:
        st.metric("Supported", supported_count)
    with col3:
        st.metric("Hallucinated", hallucinated_count)
    with col4:
        st.metric("Overall Score", result["scores"]["overall_faithfulness"])

    if result["hallucinated_spans"]:
        st.warning("⚠️ Hallucinated Spans Detected:")
        for span in result["hallucinated_spans"][:5]:
            st.write(f"- {span['span']} (confidence: {span['confidence']})")

    st.markdown("---")

    st.subheader("📊 Faithfulness Metrics")

    unsupported_count = sum(1 for s in result["span_analysis"] if s["status"] == "hallucinated")
    faithfulness_rounded = round(result["scores"]["faithfulness_score"], 2)
    evidence_coverage_rounded = round(result["scores"]["evidence_coverage_score"], 2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Faithfulness Score", faithfulness_rounded)
    with col2:
        st.metric("Unsupported Claims", unsupported_count)
    with col3:
        st.metric("Evidence Coverage", evidence_coverage_rounded)

    st.markdown("---")

    st.subheader("📚 Retrieved Documents")
    for i, ret in enumerate(result["retrieval_results"][:3], 1):
        with st.expander(f"Source {i}: Article {ret['metadata'].get('article', 'N/A')} (Score: {ret['final_score']})"):
            st.write(ret["content"][:500] + "...")

    st.markdown("---")

    st.subheader("🔬 Detailed Span Analysis")
    span_df = []
    for span in result["span_analysis"][:10]:
        span_df.append({
            "Span": span["span"][:50],
            "Confidence": span["confidence"],
            "Status": span["status"]
        })

    if span_df:
        st.dataframe(span_df, use_container_width=True)
