"""
Insurance Policy RAG System - Hugging Face Spaces
Standalone Gradio interface with integrated RAG pipeline.
"""
import os
import shutil
import gradio as gr
from typing import Dict, Any, List
import sys

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from app.rag_pipeline import (
    ingest_documents,
    retrieve_with_metadata_filter,
    format_docs_with_citations,
    extract_sources_from_docs
)
from app.config import DEFAULT_K_RESULTS, MAX_K_RESULTS

# Initialize OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INSURANCE_POLICIES_DIR = "data/insurance_policies"

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Please set it in Hugging Face Spaces secrets.")


def query_insurance(question: str) -> str:
    """Answer a question about insurance policies."""
    if not question.strip():
        return "Please enter a question."
    
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in Hugging Face Spaces secrets."
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Retrieve relevant documents
        docs = retrieve_with_metadata_filter(question, k=DEFAULT_K_RESULTS)
        
        if not docs:
            return "No relevant information found in the policy documents."
        
        # Format context
        context = format_docs_with_citations(docs)
        sources = extract_sources_from_docs(docs)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert insurance policy analyst. Answer questions based ONLY on the provided policy documents.

**CRITICAL PRIORITIZATION RULES:**

1. **Personal Policy Documents Take Priority**: If the user asks about "my policy", "am I covered", "what is my [X]", ALWAYS prioritize information from **PERSONAL POLICY DOCUMENTS** (clearly marked) over general insurance guides.

2. **Identify Document Types**: 
   - **Personal Policies**: Contain specific policy numbers, policyholder names, and exact coverage amounts - these are the user's ACTUAL policies
   - **General Guides**: Provide educational/typical information - these are NOT the user's actual policy
   - **When both types are present**: Cite personal policy information first and use general guides only for additional context

3. **Answer Specificity**:
   - For "my/am I/do I" questions: Extract SPECIFIC details from personal policy documents (policy numbers, exact limits, actual deductibles, named insureds)
   - For general questions: Use general insurance guides
   - If personal policy exists but doesn't contain the answer, state: "Your policy documents don't specify [X], but typically in insurance..."

4. **Coverage Determination**:
   - State clearly if something IS covered, IS NOT covered, or has LIMITED coverage
   - Quote exact policy language for exclusions and limitations
   - Cite specific coverage amounts and limits when available
   - Mention deductibles that apply

5. **Source Citation**: Always indicate whether information comes from the user's personal policy or general insurance knowledge

**Instructions**:
- Provide specific, accurate answers with coverage amounts, limits, deductibles, and exclusions
- Quote relevant policy text when appropriate, use quotation marks
- If information is not in the documents, say so clearly - DO NOT make up information
- For coverage questions, specify: covered/partially covered/excluded and under what conditions
- Reference policy sections, pages, or clause numbers when available
- Use clear formatting: bullet points for lists, bold for key terms
- If multiple policies apply, compare and contrast them"""),
            ("user", """Policy Documents:
{context}

Question: {question}

Provide a clear, detailed answer. If this is a personal policy question (contains "my", "I am", "am I", "do I"), prioritize personal policy documents in your answer.""")
        ])
        
        # Create chain
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | llm | StrOutputParser()
        
        # Get answer
        answer = chain.invoke({"context": context, "question": question})
        
        # Format response
        result = f"**Answer:** {answer}"
        if sources:
            result += f"\n\n📄 **Sources:** {', '.join(sources)}"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


def check_coverage(scenario: str) -> str:
    """Check if a scenario is covered by the policy."""
    if not scenario.strip():
        return "Please describe a scenario."
    
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in Hugging Face Spaces secrets."
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel, Field
        
        class CoverageResponse(BaseModel):
            is_covered: bool = Field(description="Whether the scenario is covered")
            coverage_determination: str = Field(description="Explanation of coverage determination")
            coverage_limit: str = Field(default="", description="Coverage limit if applicable")
            deductible: str = Field(default="", description="Deductible if applicable")
            policy_section: str = Field(default="", description="Relevant policy section")
        
        # Retrieve relevant documents
        docs = retrieve_with_metadata_filter(scenario, k=DEFAULT_K_RESULTS)
        
        if not docs:
            return "No relevant policy information found."
        
        context = format_docs_with_citations(docs)
        sources = extract_sources_from_docs(docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an insurance policy expert. Analyze whether a scenario is covered based on the policy documents.

**PRIORITY**: If personal policy documents are provided (marked as "PERSONAL POLICY"), base your analysis on those. General guides are for context only.

**Analysis Steps**:
1. Determine if scenario is COVERED, EXCLUDED, or PARTIALLY COVERED
2. Quote specific policy language supporting your determination
3. Identify relevant deductibles, sub-limits, or conditions
4. Cite exact policy section/clause

Return a JSON response with:
- is_covered: boolean (true if covered or partially covered, false if excluded)
- coverage_determination: detailed explanation with policy quotes and confidence level (HIGH/MEDIUM/LOW)
- coverage_limit: any applicable limits or "Not specified"
- deductible: any applicable deductible or "Not specified"
- policy_section: relevant section reference or "Not specified"

BE SPECIFIC: Use exact policy language and numbers when available."""),
            ("user", """Policy Documents:
{context}

Scenario: {scenario}

Analyze coverage and respond in JSON format.""")
        ])
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        parser = JsonOutputParser(pydantic_object=CoverageResponse)
        chain = prompt | llm | parser
        
        response = chain.invoke({"context": context, "scenario": scenario})
        
        # Format response
        is_covered = "✅ YES" if response.get("is_covered") else "❌ NO"
        result = f"## Coverage: {is_covered}\n\n"
        result += f"**Determination:** {response.get('coverage_determination', 'N/A')}\n\n"
        
        if response.get("coverage_limit"):
            result += f"**Coverage Limit:** {response.get('coverage_limit')}\n"
        if response.get("deductible"):
            result += f"**Deductible:** {response.get('deductible')}\n"
        if response.get("policy_section"):
            result += f"**Policy Section:** {response.get('policy_section')}\n"
        
        if sources:
            result += f"\n📄 **Sources:** {', '.join(sources)}"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


def compare_policies(comparison_query: str) -> str:
    """Compare aspects across policies."""
    if not comparison_query.strip():
        return "Please enter what you want to compare."
    
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in Hugging Face Spaces secrets."
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Retrieve more documents for comparison
        docs = retrieve_with_metadata_filter(comparison_query, k=MAX_K_RESULTS)
        
        if not docs:
            return "No policies found for comparison."
        
        context = format_docs_with_citations(docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an insurance policy comparison expert. Compare the requested aspect across all available policies.

**Instructions**:
1. Create a clear comparison table or structured format
2. Highlight KEY DIFFERENCES in coverage, limits, deductibles
3. Note any gaps or overlaps in coverage
4. Use specific numbers and amounts from the policies
5. If comparing personal policy to typical coverage, clearly distinguish which is which
6. If personal policy documents are available, emphasize those in the comparison

**Format your response with**:
- Clear headings for each policy (distinguish PERSONAL vs GENERAL GUIDE)
- Side-by-side comparison of key features
- Bullet points for differences
- Bold for critical distinctions
- Summary of recommendations or important findings"""),
            ("user", """Policy Documents:
{context}

Comparison Request: {query}

Provide a detailed, structured comparison.""")
        ])
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"context": context, "query": comparison_query})
        
        return f"## Policy Comparison\n\n{result}"
        
    except Exception as e:
        return f"Error: {str(e)}"


def upload_files(files) -> str:
    """Upload personal insurance policy files."""
    if not files:
        return "❌ No files selected. Please choose files to upload."
    
    try:
        # Ensure directory exists
        os.makedirs(INSURANCE_POLICIES_DIR, exist_ok=True)
        
        uploaded_count = 0
        uploaded_files = []
        
        for file in files:
            # Get filename
            filename = os.path.basename(file.name)
            destination = os.path.join(INSURANCE_POLICIES_DIR, filename)
            
            # Copy file to insurance_policies directory
            shutil.copy(file.name, destination)
            uploaded_files.append(filename)
            uploaded_count += 1
        
        result = f"## ✅ Upload Successful!\n\n"
        result += f"**Files uploaded:** {uploaded_count}\n\n"
        for fname in uploaded_files:
            result += f"- {fname}\n"
        result += f"\n💡 **Next step:** Click 'Ingest Documents' button below to process these files."
        
        return result
        
    except Exception as e:
        return f"❌ Error uploading files: {str(e)}"


def ingest_docs() -> str:
    """Trigger document ingestion."""
    if not OPENAI_API_KEY:
        return "❌ OpenAI API key not configured. Please set OPENAI_API_KEY in Hugging Face Spaces secrets."
    
    try:
        result = ingest_documents()
        
        return f"""✅ **Ingestion Complete!**

- **Documents Processed:** {result.get('documents_processed', 0)}
- **Chunks Created:** {result.get('chunks_count', 0)}
- **Policy Types Found:** {', '.join(result.get('policy_types_found', []))}

{result.get('message', '')}"""
    except Exception as e:
        return f"Error during ingestion: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title="Insurance Policy RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 Insurance Policy RAG System
    
    **✅ Pre-loaded with comprehensive insurance knowledge!**
    
    Ask questions about:
    - 🏠 Homeowners Insurance
    - 🚗 Auto Insurance  
    - 🏥 Health Insurance
    - 💼 Life Insurance
    - 🏢 Renters Insurance
    - 📖 Insurance Terms & Definitions
    
    💡 **Optional:** Upload your personal insurance policies in the Admin tab for customized advice specific to your coverage.
    
    **Note:** Set your OpenAI API key in Hugging Face Spaces secrets (environment variable `OPENAI_API_KEY`).
    """)
    
    with gr.Tabs():
        # Tab 1: Query
        with gr.TabItem("💬 Ask Questions"):
            gr.Markdown("Ask any question about your insurance policies.")
            
            question_input = gr.Textbox(
                placeholder="e.g., What is my homeowner's insurance deductible?",
                label="Your Question",
                lines=2
            )
            
            submit_btn = gr.Button("Ask", variant="primary", size="lg")
            
            answer_output = gr.Markdown(label="Answer")
            
            gr.Examples(
                examples=[
                    "What is the typical deductible for homeowners insurance?",
                    "Is flood damage covered by standard homeowners insurance?",
                    "What types of auto insurance coverage exist?",
                    "What's the difference between HMO and PPO health insurance?",
                    "How much life insurance coverage do I need?",
                ],
                inputs=question_input
            )
            
            submit_btn.click(query_insurance, question_input, answer_output)
            question_input.submit(query_insurance, question_input, answer_output)
        
        # Tab 2: Coverage Check
        with gr.TabItem("✅ Check Coverage"):
            gr.Markdown("Describe a scenario to check if it's covered by your policy.")
            
            scenario_input = gr.Textbox(
                placeholder="e.g., A tree fell on my house during a storm",
                label="Describe Your Scenario",
                lines=3
            )
            coverage_output = gr.Markdown(label="Coverage Result")
            
            check_btn = gr.Button("Check Coverage", variant="primary", size="lg")
            
            gr.Examples(
                examples=[
                    "My car was damaged in a flood",
                    "A tree fell on my house during a storm",
                    "Someone slipped and fell on my property",
                    "My laptop was stolen from my car",
                    "Lightning struck my home and damaged electronics",
                ],
                inputs=scenario_input
            )
            
            check_btn.click(check_coverage, scenario_input, coverage_output)
            scenario_input.submit(check_coverage, scenario_input, coverage_output)
        
        # Tab 3: Compare Policies
        with gr.TabItem("📊 Compare Policies"):
            gr.Markdown("Compare coverage, limits, or deductibles across your policies.")
            
            compare_input = gr.Textbox(
                placeholder="e.g., Compare deductibles across all policies",
                label="What to Compare",
                lines=2
            )
            compare_output = gr.Markdown(label="Comparison Result")
            
            compare_btn = gr.Button("Compare", variant="primary", size="lg")
            
            gr.Examples(
                examples=[
                    "Compare deductibles across all policies",
                    "Compare liability coverage limits",
                    "What are the coverage limits in each policy?",
                    "Compare exclusions across policies",
                ],
                inputs=compare_input
            )
            
            compare_btn.click(compare_policies, compare_input, compare_output)
            compare_input.submit(compare_policies, compare_input, compare_output)
        
        # Tab 4: Admin
        with gr.TabItem("⚙️ Admin"):
            gr.Markdown("""
            ### 📚 Document Management
            
            **Default Knowledge Base:**
            - ✅ Pre-loaded with general insurance guides (always available)
            - Covers homeowners, auto, health, life, renters insurance, and terminology
            
            **Personal Policies (Optional):**
            Upload your personal insurance documents for customized advice specific to YOUR coverage.
            """)
            
            # File upload section
            gr.Markdown("### 📤 Upload Personal Insurance Policies")
            
            file_upload = gr.File(
                label="Choose your insurance policy files (PDF, TXT, DOCX, MD)",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".docx", ".md"]
            )
            
            upload_btn = gr.Button("📤 Upload Files", variant="secondary", size="lg")
            upload_output = gr.Markdown(label="Upload Status")
            
            upload_btn.click(upload_files, inputs=file_upload, outputs=upload_output)
            
            gr.Markdown("---")
            
            # Ingestion section
            gr.Markdown("""
            ### 🔄 Process Documents
            Click button below to ingest/re-ingest all documents (default + personal).
            
            **For Hugging Face Spaces:** Upload documents before clicking ingest.
            """)
            
            ingest_btn = gr.Button("📥 Ingest Documents", variant="primary", size="lg")
            ingest_output = gr.Markdown(label="Ingestion Result")
            
            ingest_btn.click(ingest_docs, None, ingest_output)
            
            gr.Markdown("""
            ---
            ### 📋 Supported File Types
            - PDF (.pdf)
            - Text (.txt)
            - Word (.docx)
            - Markdown (.md)
            
            **Directories:**
            - Default docs: `data/default_insurance_docs/` (pre-loaded, no upload needed)
            - Your policies: `data/insurance_policies/` (uploaded via button above)
            
            ### How to Deploy on Hugging Face Spaces
            1. Create a new Space on Hugging Face (select Gradio SDK)
            2. Upload all files from this project (includes default docs)
            3. Set `OPENAI_API_KEY` in Settings → Repository secrets
            4. (Optional) Upload personal policies to `data/insurance_policies/`
            5. Use the Admin tab to ingest documents
            6. Start querying!
            """)
    
    gr.Markdown("""
    ---
    ### About
    This RAG (Retrieval-Augmented Generation) system uses:
    - **LangChain** for document processing and RAG pipeline
    - **ChromaDB** for vector storage
    - **OpenAI GPT-4** for natural language understanding
    - **Gradio** for the user interface
    
    Supports multiple insurance types: Homeowners, Auto, Commercial Property, Umbrella, and more.
    """)


# Auto-ingest default documents on first load
def auto_ingest_on_load():
    """Auto-ingest default documents if vector DB doesn't exist or is empty."""
    from app.config import VECTOR_DB_PATH
    chroma_db_file = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")
    
    needs_ingestion = False
    
    # Check if DB file exists
    if not os.path.exists(chroma_db_file):
        needs_ingestion = True
        print("📚 Vector database not found.")
    else:
        # Check if we can actually retrieve documents
        try:
            test_docs = retrieve_with_metadata_filter("insurance", k=1)
            if not test_docs:
                needs_ingestion = True
                print("📚 Vector database exists but is empty.")
        except Exception as e:
            needs_ingestion = True
            print(f"📚 Vector database check failed: {e}")
    
    if needs_ingestion:
        print("\n" + "="*80)
        print("📚 First-time setup: Auto-ingesting default insurance documents...")
        print("="*80)
        try:
            result = ingest_documents()
            print(f"✅ Auto-ingestion complete!")
            print(f"   - Documents: {result.get('documents_processed', 0)}")
            print(f"   - Chunks: {result.get('chunks_count', 0)}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"⚠️ Auto-ingestion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print("   You can manually ingest using the Admin tab\n")
    else:
        print("✅ Vector database ready with documents.")

if __name__ == "__main__":
    # Run auto-ingestion on startup
    if OPENAI_API_KEY:
        auto_ingest_on_load()
    else:
        print("⚠️ OPENAI_API_KEY not set. Please configure it in Hugging Face Spaces secrets.")
    
    demo.launch(server_name="0.0.0.0", server_port=7860)
