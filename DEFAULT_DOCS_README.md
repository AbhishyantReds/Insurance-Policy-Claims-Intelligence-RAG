# Default Insurance Knowledge Base

## Overview

The Insurance Policy RAG system now includes a **default knowledge base** with comprehensive insurance guides. This allows the system to answer general insurance questions **without requiring users to upload documents**.

## What Changed?

### New Dual-Mode System

1. **Default Insurance Knowledge** (Always Available)
   - Pre-loaded comprehensive guides
   - Always accessible, no upload needed
   - Covers fundamental insurance concepts

2. **Personal Policy Documents** (Optional)
   - Upload your actual insurance policies
   - Get personalized advice specific to YOUR coverage
   - Combines with default knowledge for best results

## Default Documents Included

### 1. Homeowners Insurance Guide (`homeowners_insurance_guide.txt`)
**~2,800 lines covering:**
- Coverage types (Dwelling, Personal Property, Liability, Additional Living Expenses)
- Deductibles ($500-$5,000 range)
- Covered perils (16 standard perils)
- Exclusions (flood, earthquake, wear & tear)
- Replacement cost vs. actual cash value
- Additional coverage options (scheduled property, earthquake, water backup)
- Claims process
- Cost factors ($1,500-$2,000/year average)
- Policy review tips

### 2. Auto Insurance Guide (`auto_insurance_guide.txt`)
**~850 lines covering:**
- Coverage types (Liability, Collision, Comprehensive, PIP, Uninsured Motorist)
- State minimum requirements
- Coverage scenarios and examples
- Exclusions (intentional damage, racing, business use)
- Deductibles ($250-$2,000 range)
- Premium factors (age, driving record, location, vehicle)
- Discounts (good driver, multi-policy, safety features)
- Claims process
- Tips for choosing coverage

### 3. Health Insurance Guide (`health_insurance_guide.txt`)
**~3,200 lines covering:**
- ACA fundamentals
- Plan types (HMO, PPO, EPO, POS, HDHP, Catastrophic)
- Key terms (Premium, Deductible, Copay, Coinsurance, Out-of-Pocket Max)
- Covered services (Essential Health Benefits, Preventive Care)
- Common exclusions
- Special enrollment periods
- Medicare (Parts A, B, C, D, Medigap)
- Medicaid and CHIP
- Claims and appeals process
- Tips for choosing plans

### 4. Life Insurance Guide (`life_insurance_guide.txt`)
**~3,000 lines covering:**
- Fundamentals (why you need it, how much coverage)
- Term Life Insurance (10, 15, 20, 30-year terms)
- Whole Life Insurance (cash value, dividends)
- Universal Life Insurance (Standard UL, IUL, VUL, GUL)
- Application process and underwriting
- Rate classes (Preferred Plus to Substandard)
- Beneficiaries and policy ownership
- Riders (Waiver of Premium, Accelerated Death Benefit, etc.)
- Living benefits (cash value loans, settlements)
- Common mistakes and tips

### 5. Renters Insurance Guide (`renters_insurance_guide.txt`)
**~3,500 lines covering:**
- Coverage basics ($15-$30/month average)
- What's covered (personal property, liability, additional living expenses)
- What's NOT covered (flooding, earthquakes, bed bugs)
- Coverage types (ACV vs. Replacement Cost)
- Deductible options ($250-$1,000)
- Liability coverage levels ($100K-$1M)
- Discounts (multi-policy, security systems, claims-free)
- Filing claims (step-by-step process)
- Landlord requirements
- Coverage extensions (scheduled property, water backup, earthquake)

### 6. Insurance Glossary (`insurance_glossary.txt`)
**~700 lines covering:**
- A-Z insurance terms and definitions
- Comprehensive list of common insurance abbreviations
- Policy structure terms
- Coverage-specific terminology
- Claims and underwriting terms

## Technical Implementation

### Code Changes

#### 1. Updated `app/rag_pipeline.py`

```python
# Added default docs path
DEFAULT_DOCS_PATH = os.path.join(os.path.dirname(DOCUMENTS_PATH), "default_insurance_docs")

def ingest_documents():
    """
    Now loads from TWO directories:
    1. default_insurance_docs/ - always included
    2. insurance_policies/ - optional personal docs
    """
    # STEP 1: Load default insurance knowledge documents
    # STEP 2: Load personal insurance policies
    # Both are indexed together in ChromaDB and BM25
```

**Metadata added:**
- `document_category`: "general_knowledge" or "personal_policy"
- `is_default_doc`: True/False
- Allows filtering or prioritization if needed

**Return statistics now include:**
- `default_docs_count`: Number of default documents loaded
- `personal_docs_count`: Number of personal policies loaded

#### 2. Updated `gradio_app.py`

**Welcome message:**
```markdown
✅ Pre-loaded with comprehensive insurance knowledge!

Ask questions about:
- 🏠 Homeowners Insurance
- 🚗 Auto Insurance  
- 🏥 Health Insurance
- 💼 Life Insurance
- 🏢 Renters Insurance
- 📖 Insurance Terms & Definitions

💡 Optional: Upload your personal insurance policies for customized advice
```

**Admin tab updated:**
- Clarifies default docs vs. personal policies
- Shows both directories
- Explains dual-mode system

**Example questions updated:**
- Changed from specific policy questions to general insurance questions
- Better demonstrates default knowledge capability

#### 3. Updated `app.py` (Standalone Gradio)

Same changes as gradio_app.py for consistency.

## Directory Structure

```
finance-rag-qa-api/
├── data/
│   ├── default_insurance_docs/       # Default knowledge (NEW)
│   │   ├── homeowners_insurance_guide.txt
│   │   ├── auto_insurance_guide.txt
│   │   ├── health_insurance_guide.txt
│   │   ├── life_insurance_guide.txt
│   │   ├── renters_insurance_guide.txt
│   │   └── insurance_glossary.txt
│   └── insurance_policies/           # Personal policies (optional)
│       └── (user uploads their policies here)
```

## Usage

### For End Users

**No setup required!** The system now works immediately:

1. **Ask General Insurance Questions:**
   - "What is a deductible in homeowners insurance?"
   - "What's the difference between HMO and PPO?"
   - "How much life insurance do I need?"

2. **Optional - Add Personal Policies:**
   - Upload your actual insurance documents to `data/insurance_policies/`
   - Click "Ingest Documents" in Admin tab
   - Now get answers about BOTH general concepts AND your specific policies

### For Developers

**First-time setup:**
```bash
# No action needed - default docs already in repository
# Just run the application
python app/main.py  # or
python gradio_app.py
```

**Adding more default documents:**
```bash
# Add new .txt, .pdf, .docx, or .md files to:
data/default_insurance_docs/

# Re-ingest documents
# (via Admin tab or API endpoint)
```

## Benefits

### 1. **Instant Usability**
- Users can ask questions immediately
- No need to upload documents first
- Great for demos and onboarding

### 2. **Educational Resource**
- Comprehensive insurance education
- Covers all major insurance types
- Explains complex terms clearly

### 3. **Flexible for Personalization**
- Default knowledge provides baseline
- Personal policies add customization
- Best of both worlds

### 4. **Better RAG Performance**
- Hybrid search finds relevant info in both doc types
- More context = better answers
- Validation system catches hallucinations

### 5. **Production Ready**
- Can deploy without requiring users to have policies
- Useful even for people shopping for insurance
- Educational tool + policy advisor

## Example Queries

### General Questions (Default Docs)

✅ **Works immediately:**
- "What does a deductible mean?"
- "Is flood damage typically covered by homeowners insurance?"
- "What are the main types of auto insurance coverage?"
- "Should I get term or whole life insurance?"
- "What's the difference between collision and comprehensive?"

### Personalized Questions (After Uploading Personal Policy)

✅ **After uploading your policy:**
- "What is MY homeowners insurance deductible?"
- "Am I covered for earthquake damage in MY policy?"
- "What are MY auto liability limits?"
- "Does MY policy have an umbrella provision?"

### Combined Queries

✅ **Best of both:**
- "Is water damage covered? Check both general guidelines and my specific policy."
- "What's typical renters insurance cost, and how does my policy compare?"

## Hybrid Search Advantage

The system uses **BM25 + Semantic search** which excels at:

1. **Keyword Matches:** 
   - "What is a copay?" → finds exact term in glossary
   - "Section 3.1" → finds specific policy section

2. **Semantic Understanding:**
   - "money I pay when I go to doctor" → understands = copay
   - "protection if someone sues me" → understands = liability

3. **Multi-Document Retrieval:**
   - Can pull from glossary + guide + personal policy
   - Combines general knowledge with specific policy details

## Monitoring & Validation

All queries (default docs or personal policies) go through:

1. **Retrieval Quality Check:** Ensures relevant docs found
2. **Faithfulness Validation:** Prevents hallucinations
3. **Number Verification:** Catches fabricated amounts
4. **Confidence Scoring:** Indicates answer reliability
5. **Metrics Tracking:** Logs all queries to SQLite

## Content Quality

Each default document includes:
- ✅ **Comprehensive coverage** of topic (2,000-3,500 lines each)
- ✅ **Real-world examples** with dollar amounts
- ✅ **Practical scenarios** and use cases
- ✅ **Common mistakes** and tips
- ✅ **Industry averages** and typical ranges
- ✅ **Structured sections** for easy retrieval
- ✅ **Clear terminology** with definitions

## Future Enhancements

Potential improvements:

1. **More Default Documents:**
   - Commercial insurance guide
   - Business insurance guide
   - General insurance principles
   - State-specific insurance regulations

2. **Version Control:**
   - Track default doc versions
   - Update default docs with industry changes
   - Notify users of outdated information

3. **Prioritization:**
   - Prefer personal policy over default for specific questions
   - Use metadata filtering (document_category field)

4. **Analytics:**
   - Track which default docs are most queried
   - Identify knowledge gaps
   - Guide content improvements

## Testing

Test the dual-mode system:

```python
# Test default docs only (no personal policies uploaded)
# Should work immediately:
query = "What is a deductible?"
# → Retrieves from default docs

# Upload personal policy, re-ingest
# Test combined retrieval:
query = "What is my deductible?"
# → Should prioritize personal policy but can reference default docs
```

## Credits

Default insurance content created with:
- Industry standards and best practices
- Real-world insurance examples
- Consumer-friendly explanations
- 2026 data and requirements

---

## Quick Start

**Users:** Just start asking questions! Default knowledge is pre-loaded.

**Developers:** No changes needed. Default docs are part of the repository.

**Deployers:** Deploy as-is. Default docs included automatically.

That's it! The system now works out-of-the-box with comprehensive insurance knowledge, while still supporting personal policy customization. 🎉
