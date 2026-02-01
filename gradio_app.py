"""
Insurance Policy RAG - Gradio Interface
Simple chat interface for querying insurance policies.
"""
import os
import shutil
import gradio as gr
import requests

# API Base URL
API_URL = "http://127.0.0.1:8000"
INSURANCE_POLICIES_DIR = "data/insurance_policies"


def query_insurance(question: str, history: list) -> str:
    """Send question to the RAG API and get response."""
    if not question.strip():
        return "Please enter a question."
    
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer found.")
            sources = data.get("sources", [])
            
            # Format response
            result = f"**Answer:** {answer}"
            
            if sources:
                result += f"\n\n📄 **Sources:** {', '.join(sources)}"
            
            return result
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure the server is running on http://127.0.0.1:8000"
    except Exception as e:
        return f"Error: {str(e)}"


def check_coverage(scenario: str) -> str:
    """Check if a scenario is covered."""
    if not scenario.strip():
        return "Please describe a scenario."
    
    try:
        response = requests.post(
            f"{API_URL}/check-coverage",
            json={"scenario": scenario},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Format structured response
            is_covered = "✅ YES" if data.get("is_covered") else "❌ NO"
            result = f"## Coverage: {is_covered}\n\n"
            result += f"**Determination:** {data.get('coverage_determination', 'N/A')}\n\n"
            
            if data.get("coverage_limit"):
                result += f"**Coverage Limit:** {data.get('coverage_limit')}\n"
            if data.get("deductible"):
                result += f"**Deductible:** {data.get('deductible')}\n"
            if data.get("policy_section"):
                result += f"**Policy Section:** {data.get('policy_section')}\n"
            
            sources = data.get("sources", [])
            if sources:
                result += f"\n📄 **Sources:** {', '.join(sources)}"
            
            return result
        else:
            return f"Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure the server is running."
    except Exception as e:
        return f"Error: {str(e)}"


def compare_policies(comparison_query: str) -> str:
    """Compare policies."""
    if not comparison_query.strip():
        return "Please enter what you want to compare."
    
    try:
        response = requests.post(
            f"{API_URL}/compare-policies",
            json={"comparison_query": comparison_query},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            result = f"## {data.get('comparison_type', 'Comparison')}\n\n"
            
            items = data.get("comparison_items", [])
            if items:
                result += "| Policy Type | Value | Section |\n"
                result += "|-------------|-------|--------|\n"
                for item in items:
                    result += f"| {item.get('policy_type', 'N/A')} | {item.get('value', 'N/A')} | {item.get('section', 'N/A')} |\n"
            
            result += f"\n**Summary:** {data.get('summary', 'N/A')}"
            
            return result
        else:
            return f"Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure the server is running."
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


def ingest_documents() -> str:
    """Trigger document ingestion."""
    try:
        response = requests.post(f"{API_URL}/ingest", timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            return f"""✅ **Ingestion Complete!**
            
- **Documents Processed:** {data.get('documents_processed', 0)}
- **Chunks Created:** {data.get('chunks_count', 0)}
- **Policy Types Found:** {', '.join(data.get('policy_types_found', []))}

{data.get('message', '')}"""
        else:
            return f"Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure the server is running on http://127.0.0.1:8000"
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio Interface
with gr.Blocks(title="Insurance Policy RAG") as demo:
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
    """)
    
    with gr.Tabs():
        # Tab 1: Query
        with gr.TabItem("💬 Ask Questions"):
            gr.Markdown("Ask any question about your insurance policies.")
            
            question_input = gr.Textbox(
                placeholder="e.g., What is my health insurance sum insured?",
                label="Your Question",
                lines=2
            )
            
            submit_btn = gr.Button("Ask", variant="primary")
            
            answer_output = gr.Markdown(label="Answer")
            
            gr.Examples(
                examples=[
                    "What is the deductible for homeowners insurance?",
                    "Is flood damage covered by standard homeowners insurance?",
                    "What types of auto insurance coverage are there?",
                    "What is the difference between HMO and PPO health insurance?",
                    "How much life insurance coverage do I need?",
                ],
                inputs=question_input
            )
            
            def get_answer(question):
                return query_insurance(question, [])
            
            submit_btn.click(get_answer, question_input, answer_output)
            question_input.submit(get_answer, question_input, answer_output)
        
        # Tab 2: Coverage Check
        with gr.TabItem("✅ Check Coverage"):
            gr.Markdown("Describe a scenario to check if it's covered by your policy.")
            
            scenario_input = gr.Textbox(
                placeholder="e.g., My car was damaged in a flood",
                label="Describe Your Scenario",
                lines=3
            )
            coverage_output = gr.Markdown(label="Coverage Result")
            
            check_btn = gr.Button("Check Coverage", variant="primary")
            
            gr.Examples(
                examples=[
                    "My car was damaged in a flood",
                    "I need hospitalization for heart surgery",
                    "Fire damaged my house",
                    "My laptop was stolen from home",
                ],
                inputs=scenario_input
            )
            
            check_btn.click(check_coverage, scenario_input, coverage_output)
        
        # Tab 3: Compare Policies
        with gr.TabItem("📊 Compare Policies"):
            gr.Markdown("Compare coverage, limits, or deductibles across your policies.")
            
            compare_input = gr.Textbox(
                placeholder="e.g., Compare deductibles across all policies",
                label="What to Compare",
                lines=2
            )
            compare_output = gr.Markdown(label="Comparison Result")
            
            compare_btn = gr.Button("Compare", variant="primary")
            
            gr.Examples(
                examples=[
                    "Compare deductibles across all policies",
                    "Compare sum insured amounts",
                    "What are the coverage limits in each policy",
                ],
                inputs=compare_input
            )
            
            compare_btn.click(compare_policies, compare_input, compare_output)
        
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
            
            upload_btn = gr.Button("📤 Upload Files", variant="secondary")
            upload_output = gr.Markdown(label="Upload Status")
            
            upload_btn.click(upload_files, inputs=file_upload, outputs=upload_output)
            
            gr.Markdown("---")
            
            # Ingestion section
            gr.Markdown("""
            ### 🔄 Process Documents
            Click button below to ingest/re-ingest all documents (default + personal).
            """)
            
            ingest_btn = gr.Button("📥 Ingest Documents", variant="primary")
            ingest_output = gr.Markdown(label="Ingestion Result")
            
            ingest_btn.click(ingest_documents, None, ingest_output)
            
            gr.Markdown("""
            ---
            ### 📋 Supported File Types
            - PDF (.pdf)
            - Text (.txt)
            - Word (.docx)
            - Markdown (.md)
            
            ### 📂 Directories
            - **Default docs:** `data/default_insurance_docs/` (pre-loaded, no upload needed)
            - **Your policies:** `data/insurance_policies/` (uploaded via button above)
            """)


if __name__ == "__main__":
    # Use server_name="0.0.0.0" to allow external access, or None for localhost only
    demo.launch(share=False)
