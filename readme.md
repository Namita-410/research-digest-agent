Markdown
# Autonomous Research Digest Agent

An intelligent, graph-based autonomous agent that ingests multiple sources (URLs and local file folders), extracts evidence-backed claims, deduplicates them using mathematical vector clustering, and generates a structured research brief.


Built with **Python, LangGraph, LangChain, Scikit-Learn, Matplotlib, and Groq (Llama 3)**.

##  Run Instructions

### 1. Environment Setup
Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
2. Install Dependencies
Bash
python -m pip install -r requirements.txt
3. Set API Key
Get a free API key from Groq.

Bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-api-key"

# Mac/Linux
export GROQ_API_KEY="your-api-key"

4. Run the Agent & Tests
Run Agent: python graph_agent.py

Run Tests: pytest test_agent.py

## Architecture & Engineering Decisions
How the agent processes sources step-by-step
The agent is designed as a LangGraph State Machine rather than a linear script, allowing for extreme scalability and error handling. It consists of four nodes:

Ingest Node: Iterates through a mix of URLs and local file folders using glob. It safely catches and logs fetch failures (e.g., Cloudflare 403 errors, 404 dead links) without crashing the pipeline.

Extract Node (with Caching): [Stretch Goal Unlocked] Before hitting the LLM, the agent checks local_cache.json. If the URL/file was processed previously, it instantly loads the data, saving API tokens and time. If new, it passes the text to the LLM via LangChain Structured Outputs.

Group Node (Math-Based Clustering): [Stretch Goal Unlocked] To prevent LLM hallucinations at scale, all extracted claims are converted into TF-IDF vectors. A Cosine Similarity matrix calculates the distance between every claim. The mathematical clusters are then visualized via PCA and saved as a 2D scatter plot before being sent to the LLM for final thematic summarization.

Output Node: Python logic formats the synthesized State data into digest.md and dumps the raw mappings into sources.json.

How claims are grounded
Claims are strictly grounded by forcing the LLM (via system prompting and strict Pydantic schema validation) to return an exact, word-for-word quote from the source text alongside every extracted claim. If the LLM cannot find a quote, it cannot fulfill the schema contract. Furthermore, [Stretch Goal Unlocked] the schema forces the LLM to assign a Confidence Score (1-10) grading how strongly the quote proves the claim.

How deduplication/grouping works
[Stretch Goal Unlocked: Configurable Grouping Threshold]. Instead of relying on rigid string matching or prompting an LLM to guess which claims are similar, the agent utilizes pure mathematics. It generates TfidfVectorizer embeddings for every claim. It uses cosine_similarity from scikit-learn to programmatically cluster claims that cross a highly configurable threshold (currently set to 0.50).

One Limitation:
The current extraction method caps at 8,000 characters per webpage to stay within context windows and minimize inference time. For highly dense, 50-page research PDFs, valuable claims at the end of the document might be truncated. Additionally, using standard requests means heavy client-side JavaScript pages or strict anti-bot firewalls might return empty text.

One Improvement with more time:

With more time, I would implement two things:

Autonomous Web Search: Integrate a tool like Tavily or the DuckDuckGo API as a preliminary LangGraph node. Instead of hard-coding the input URLs, the user could simply type a topic, and the agent would autonomously search the web, scrape the 5 best sources itself, and then run the digest pipeline.

Headless Browsing: Replace the requests library with Playwright or Selenium to bypass Cloudflare blocks and render JavaScript-heavy pages before scraping.