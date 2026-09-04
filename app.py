import streamlit as st

from ingestion import process_file
from vectorstore import add_chunks, query, list_sources
from research import research_company_cached
from llm import generate
import prompts

st.set_page_config(page_title="RAG Sales Assistant", layout="wide")

# --- session state ---
if "org_id" not in st.session_state:
    st.session_state.org_id = "demo_org"

st.sidebar.title("Workspace")
st.session_state.org_id = st.sidebar.text_input("Org ID", value=st.session_state.org_id,
                                                   help="Separate orgs = separate document sets. Use this to demo multi-tenancy.")

st.sidebar.divider()
st.sidebar.subheader("1. Upload your materials")
doc_type = st.sidebar.selectbox("Document type", ["pricing", "case_study", "template", "other"])
uploaded = st.sidebar.file_uploader("PDF, DOCX, PPTX, or TXT", type=["pdf", "docx", "pptx", "txt", "md"],
                                     accept_multiple_files=True)

if uploaded and st.sidebar.button("Ingest documents"):
    total = 0
    with st.sidebar.status("Processing..."):
        for f in uploaded:
            chunks = process_file(f.read(), f.name, doc_type)
            total += add_chunks(st.session_state.org_id, chunks)
    st.sidebar.success(f"Added {total} chunks from {len(uploaded)} file(s).")

sources = list_sources(st.session_state.org_id)
if sources:
    st.sidebar.caption(f"Indexed sources ({len(sources)}): " + ", ".join(sources))
else:
    st.sidebar.caption("No documents indexed yet for this org.")

st.title("RAG-powered sales content generator")
st.caption("Company-specific cold-call scripts, cold emails, pricing estimates, and deliverable matching — grounded in your own uploaded documents.")

tab1, tab2, tab3, tab4 = st.tabs(["📞 Cold-call script", "✉️ Cold email", "💰 Pricing estimate", "📦 Deliverable match"])


def show_retrieved(chunks):
    with st.expander(f"Retrieved context ({len(chunks)} chunks)"):
        for c in chunks:
            st.markdown(f"**{c['source']}** ({c['doc_type']}) — distance {c['distance']:.3f}")
            st.text(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))


with tab1:
    col1, col2 = st.columns(2)
    company = col1.text_input("Target company name", key="cc_company")
    notes = col2.text_input("Notes for the rep (optional)", key="cc_notes")
    if st.button("Generate cold-call script", type="primary"):
        if not company:
            st.warning("Enter a target company name.")
        else:
            with st.spinner("Researching company and retrieving context..."):
                research = research_company_cached(company)
                chunks = query(st.session_state.org_id, f"value proposition case studies for {company}", n_results=5)
            with st.spinner("Generating..."):
                sys_p, user_p = prompts.cold_call_script(chunks, company, research, notes)
                result = generate(sys_p, user_p)
            st.markdown(result)
            show_retrieved(chunks)

with tab2:
    col1, col2 = st.columns(2)
    company_e = col1.text_input("Target company name", key="ce_company")
    notes_e = col2.text_input("Notes (optional)", key="ce_notes")
    if st.button("Generate cold email", type="primary"):
        if not company_e:
            st.warning("Enter a target company name.")
        else:
            with st.spinner("Researching company and retrieving context..."):
                research = research_company_cached(company_e)
                chunks = query(st.session_state.org_id, f"value proposition case studies for {company_e}", n_results=5)
            with st.spinner("Generating..."):
                sys_p, user_p = prompts.cold_email_script(chunks, company_e, research, notes_e)
                result = generate(sys_p, user_p)
            st.markdown(result)
            show_retrieved(chunks)

with tab3:
    scope = st.text_area("Describe the project scope the client is asking about", key="pe_scope")
    if st.button("Generate pricing estimate", type="primary"):
        if not scope:
            st.warning("Describe the project scope first.")
        else:
            with st.spinner("Retrieving pricing data..."):
                chunks = query(st.session_state.org_id, scope, n_results=6, doc_type="pricing")
            if not chunks:
                st.error("No pricing documents indexed for this org yet. Upload a rate card first.")
            else:
                with st.spinner("Generating..."):
                    sys_p, user_p = prompts.pricing_estimate(chunks, scope)
                    result = generate(sys_p, user_p, temperature=0.2)
                st.markdown(result)
                show_retrieved(chunks)

with tab4:
    need = st.text_area("Describe what the client is looking for", key="dm_need")
    if st.button("Find best-match deliverable", type="primary"):
        if not need:
            st.warning("Describe the client's need first.")
        else:
            with st.spinner("Retrieving deliverables..."):
                chunks = query(st.session_state.org_id, need, n_results=5)
            with st.spinner("Generating..."):
                sys_p, user_p = prompts.deliverable_match(chunks, need)
                result = generate(sys_p, user_p, temperature=0.3)
            st.markdown(result)
            show_retrieved(chunks)

st.divider()
st.caption("Free-tier stack: sentence-transformers (embeddings) + ChromaDB (vector store) + Groq/Gemini (LLM) + Tavily (company research).")
