import os
import glob
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- The highly stable Math libraries ---

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
# 1. PYDANTIC MODELS (STRETCH GOAL 1: Confidence Scores)

class Claim(BaseModel):
    claim: str = Field(description="The extracted key claim or fact.")
    quote: str = Field(description="An exact, word-for-word quote from the text that proves the claim.")
    # NEW: The LLM must now rate its own evidence!
    confidence_score: int = Field(description="Score from 1 to 10 indicating how strongly the quote proves the claim.")

class SourceExtraction(BaseModel):
    claims: List[Claim] = Field(description="List of claims extracted from the source.")

class GroupedClaim(BaseModel):
    theme: str = Field(description="The broad theme or category of the claims.")
    summary: str = Field(description="A concise summary of the claims in this group.")
    supporting_sources: List[str] = Field(description="List of source URLs/files that support this theme.")
    conflicting_viewpoints: Optional[str] = Field(description="Any conflicting viewpoints found. Leave null if none.")

class DigestStructure(BaseModel):
    groups: List[GroupedClaim] = Field(description="Grouped themes synthesized from all claims.")


# 2. THE GRAPH STATE & CACHE HELPERS

class AgentState(TypedDict):
    topic: str               
    urls: List[str]
    scraped_data: List[Dict[str, Any]]
    extracted_claims: Dict[str, Any]
    grouped_data: List[Dict[str, Any]]

CACHE_FILE = "local_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)


# 3. GRAPH NODES

def ingest_sources(state: AgentState) -> dict:
    print("-> Fetching sources...")
    scraped_data = []
    
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for source in state["urls"]:
        try:
            if source.startswith("http"):
                response = requests.get(source, headers=browser_headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                text = " ".join(soup.stripped_strings)
                title = soup.title.string if soup.title else "Unknown Web Title"
            else:
                with open(source, "r", encoding="utf-8") as f:
                    text = f.read()
                title = os.path.basename(source)

            scraped_data.append({
                "url": source,
                "title": title,
                "content": text[:8000], 
                "status": "success"
            })
            
        except Exception as e:
            print(f"Failed to fetch {source}: {e}")
            scraped_data.append({"url": source, "status": "failed", "error": str(e), "content": "", "title": "Failed"})
            
    return {"scraped_data": scraped_data}

def extract_claims(state: AgentState) -> dict:
    """Node 2: Extracts claims with Caching (STRETCH GOAL 2: Ability to Re-run)"""
    print("-> Extracting claims (checking cache first)...")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(SourceExtraction)
    
    extracted_dict = {}
    cache = load_cache() # Load memory of past runs
    
    for doc in state["scraped_data"]:
        url = doc["url"]
        
        # Check if we already did the hard work for this exact URL!
        if doc["status"] == "success" and url in cache:
            print(f"   [CACHE HIT] Instantly loaded claims for: {url}")
            extracted_dict[url] = cache[url]
            continue
            
        if doc["status"] == "success" and len(doc["content"]) > 50:
            print(f"   [API CALL] Extracting new claims for: {url}")
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are a research assistant. Extract 3 claims regarding {state['topic']}. Provide exact quotes and a confidence score (1-10)."),
                ("user", "Text:\n{text}")
            ])
            try:
                response = (prompt | structured_llm).invoke({"text": doc["content"]})
                extracted_dict[url] = {
                    "title": doc["title"],
                    "claims": [c.model_dump() for c in response.claims]
                }
                # Save newly extracted data to cache so we never process it twice
                cache[url] = extracted_dict[url] 
            except Exception as e:
                extracted_dict[url] = {"title": doc["title"], "claims": [], "error": "Extraction failed"}
        else:
            extracted_dict[url] = {"title": doc["title"], "claims": [], "error": doc.get("error", "No content")}
            
    save_cache(cache)
    return {"extracted_claims": extracted_dict}

def group_claims(state: AgentState) -> dict:
    """Node 3: Embeddings, Cosine Similarity, and Visualization"""
    print("-> Grouping claims & generating visualization...")
    
    all_claims = []
    for url, data in state["extracted_claims"].items():
        if "claims" in data:
            for c in data["claims"]:
                all_claims.append({
                    "url": url, 
                    "claim_text": c["claim"], 
                    "quote": c["quote"],
                    "score": c.get("confidence_score", 0)
                })
                
    if not all_claims:
        return {"grouped_data": []}

    # --- CUSTOM HUGGING FACE API WRAPPER ---
    claim_texts = [c["claim_text"] for c in all_claims]
    
    # We call the NEW Hugging Face Router API with their officially supported free-tier model!
    api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    print("   [API CALL] Generating Hugging Face embeddings in the cloud...")
    response = requests.post(api_url, headers=headers, json={"inputs": claim_texts})
    
    if response.status_code == 200:
        vectors = response.json() 
    else:
        print(f"   [WARNING] Hugging Face API failed ({response.status_code}): {response.text}")
        return {"grouped_data": []}
    
    # Calculate Cosine Similarity on the Hugging Face vectors
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(vectors)
    
    SIMILARITY_THRESHOLD = 0.70
    clusters = []
    # ... (the rest of the grouping loop remains exactly the same!)
    cluster_labels = [-1] * len(all_claims)
    visited = set()
    cluster_id = 0
    
    for i in range(len(all_claims)):
        if i in visited:
            continue
        current_cluster = [all_claims[i]]
        cluster_labels[i] = cluster_id
        visited.add(i)
        
        for j in range(i + 1, len(all_claims)):
            if j not in visited and sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                current_cluster.append(all_claims[j])
                cluster_labels[j] = cluster_id
                visited.add(j)
                
        clusters.append(current_cluster)
        cluster_id += 1
        
    print(f"-> Mathematically grouped {len(all_claims)} claims into {len(clusters)} clusters.")
    
    # --- STRETCH GOAL 3: Visualization ---
    try:
        pca = PCA(n_components=2) # Squashes high-dimensional vectors to 2D
        reduced_vectors = pca.fit_transform(vectors)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_labels, cmap='tab20', s=100)
        plt.title("Claim Clustering (Cosine Similarity)")
        plt.savefig("cluster_visualization.png")
        plt.close()
        print("-> [SUCCESS] Saved cluster_visualization.png")
    except Exception as e:
        print(f"-> Could not generate visualization: {e}")
    # -------------------------------------

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(DigestStructure)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "I have programmatically clustered related claims. For each cluster, give it a broad theme name, write a concise summary, list the supporting URLs, and explicitly state any conflicting viewpoints."),
        ("user", "Here are the clustered claims:\n{data}")
    ])
    
    try:
        response = (prompt | structured_llm).invoke({"data": json.dumps(clusters)})
        grouped_data = [g.model_dump() for g in response.groups]
    except Exception as e:
        grouped_data = []
        
    return {"grouped_data": grouped_data}

def generate_outputs(state: AgentState) -> dict:
    print("-> Generating digest.md and sources.json...")
    
    with open("sources.json", "w", encoding="utf-8") as f:
        json.dump(state["extracted_claims"], f, indent=4)
        
    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(f"# Research Digest: {state['topic']}\n\n")
        for group in state["grouped_data"]:
            f.write(f"## {group['theme']}\n")
            f.write(f"**Summary:** {group['summary']}\n\n")
            f.write(f"**Supporting Sources:** {', '.join(group['supporting_sources'])}\n\n")
            if group.get('conflicting_viewpoints'):
                f.write(f"**Conflicting Viewpoints:** {group['conflicting_viewpoints']}\n\n")
            f.write("---\n")
            
    return {}

def run_agent(topic: str, urls: List[str]):
    print(f"Building LangGraph Pipeline for topic: '{topic}'...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("ingest", ingest_sources)
    workflow.add_node("extract", extract_claims)
    workflow.add_node("group", group_claims)
    workflow.add_node("output", generate_outputs)
    
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "group")
    workflow.add_edge("group", "output")
    workflow.add_edge("output", END)
    
    app = workflow.compile()
    app.invoke({"topic": topic, "urls": urls})
    print("\nDone! Check digest.md, sources.json, and cluster_visualization.png.")

if __name__ == "__main__":
    research_topic = "Remote Work and Productivity"
    
    sample_urls = [     "https://www.imf.org/en/blogs/articles/2024/01/14/ai-will-transform-the-global-economy-lets-make-sure-it-benefits-humanity",
"https://hbr.org/2024/11/research-how-gen-ai-is-already-impacting-the-labor-market",
"https://www.aiprm.com/ai-replacing-jobs-statistics/",
"https://budgetlab.yale.edu/research/evaluating-impact-ai-labor-market-current-state-affairs",
"https://www.dallasfed.org/research/economics/2025/0603",
"https://www.nu.edu/blog/ai-job-statistics/",
"https://www.nexford.edu/insights/how-will-ai-affect-jobs",
        
        "https://en.wikipedia.org/wiki/Remote_work",
        "https://about.gitlab.com/company/culture/all-remote/guide/",
        "https://buffer.com/state-of-remote-work/2023",
        "https://www.pewresearch.org/social-trends/2023/03/30/how-americans-view-their-jobs/",
        "https://www.technologyreview.com/2023/11/10/1083161/what-is-generative-ai-and-why-is-it-so-popular/"
    
        ]
    
    # Process the local folder we created earlier
    folder_path = "./research_files"
    local_files = glob.glob(f"{folder_path}/*.txt") + glob.glob(f"{folder_path}/*.html")
    
    all_inputs = sample_urls + local_files
    run_agent(research_topic, all_inputs)