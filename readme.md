# Autonomous Research Digest Agent

An intelligent, graph-based autonomous agent that ingests multiple sources (URLs and local file folders), extracts evidence-backed claims, deduplicates them using Hugging Face semantic AI clustering, and generates a structured research brief.


Built with **Python, LangGraph, LangChain, Hugging Face, Scikit-Learn (Cosine Similarity), Matplotlib, and Groq (Llama 3)**.

##  Run Instructions

### 1. Environment Setup
Create and activate a virtual environment:

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate

**Mac/Linux:**
python3 -m venv venv
source venv/bin/activate


2. Install Dependencies
Bash
python -m pip install -r requirements.txt
3. Set API Keys
Get free API keys from Groq and Hugging Face.

Windows (PowerShell):

Bash
$env:GROQ_API_KEY="your-groq-key"
$env:HUGGINGFACEHUB_API_TOKEN="your-hf-key"
Mac/Linux:

Bash
export GROQ_API_KEY="your-groq-key"
export HUGGINGFACEHUB_API_TOKEN="your-hf-key"


4. Run the Agent & Tests
Run Agent: python graph_agent.py

Run Tests: pytest test_agent.py

###Architecture & Engineering Decisions:
# How the agent processes sources step-by-step:
The agent is designed as a LangGraph State Machine, allowing for extreme scalability and error handling. It consists of four nodes:

Ingest Node: Iterates through a mix of URLs and local file folders using glob. It safely catches and logs fetch failures (e.g., Cloudflare 403 errors, 404 dead links) without crashing the pipeline.

Extract Node (with Caching): [Stretch Goal Unlocked] Before hitting the LLM, the agent checks local_cache.json. If the URL/file was processed previously, it instantly loads the data, saving API tokens and time. If new, it passes the text to the LLM via LangChain Structured Outputs.

Group Node (Semantic AI Clustering): [Stretch Goal Unlocked] All extracted claims are sent to the Hugging Face Router API (BAAI/bge-small-en-v1.5). A Cosine Similarity matrix calculates the semantic distance between every claim. The mathematical clusters are then visualized via PCA and saved as a 2D scatter plot before being sent to the LLM for final thematic summarization.

Output Node: Python logic formats the synthesized State data into digest.md and dumps the raw mappings into sources.json.

How claims are grounded:

Claims are strictly grounded by forcing the LLM (via system prompting and strict Pydantic schema validation) to return an exact, word-for-word quote from the source text alongside every extracted claim. If the LLM cannot find a quote, it cannot fulfill the schema contract. Furthermore, [Stretch Goal Unlocked] the schema forces the LLM to assign a Confidence Score (1-10) grading how strongly the quote proves the claim.

How deduplication/grouping works:
[Stretch Goal Unlocked: Configurable Grouping Threshold]. Rather than relying on rigid string matching, the agent utilizes a Hugging Face Semantic Embedding Model.

The Engineering Pivot: Initially, LangChain's local Hugging Face wrappers crashed due to a known PyTorch C++ dependency bug on Windows/Python 3.13. To bypass this local environment failure, I engineered a custom API call directly to the Hugging Face Router API.

This allows the agent to understand meaning rather than just counting words. Claims with a semantic Cosine Similarity of >= 0.70 are mathematically clustered together.

###One Limitation:
The current extraction method caps at 8,000 characters per webpage to stay within context windows and minimize inference time. For highly dense, 50-page research PDFs, valuable claims at the end of the document might be truncated. Additionally, using standard requests means heavy client-side JavaScript pages or strict anti-bot firewalls might return empty text.

### One Improvement with more time:

With more time, I would implement:

Autonomous Web Search: Integrate a tool like Tavily as a preliminary LangGraph node to autonomously search the web and scrape the 5 best sources dynamically before running the digest pipeline.

Headless Browsing: Replace the requests library with Playwright or Selenium to bypass Cloudflare blocks and render JavaScript-heavy pages before scraping.