import pytest
from graph_agent import ingest_sources, group_claims, AgentState

def test_empty_unreachable_source():
    """Test 1: Empty/unreachable source handling safely."""
    mock_state = {"urls": ["https://this-website-does-not-exist-12345.com"]}
    result = ingest_sources(mock_state)
    scraped = result["scraped_data"][0]
    
    assert scraped["status"] == "failed"
    assert "error" in scraped
    assert scraped["content"] == ""

def test_deduplication():
    """Test 2: Deduplication using Hugging Face Semantic Embeddings."""
    # Notice how these share NO exact keywords ("Remote work increases productivity" vs "Working from home boosts employee efficiency").
    # The Hugging Face AI understands they mean the same thing anyway!
    mock_state = {
        "extracted_claims": {
            "source1.com": {"claims": [{"claim": "Remote work increases productivity.", "quote": "..."}]},
            "source2.com": {"claims": [{"claim": "Working from home boosts employee efficiency.", "quote": "..."}]}
        },
        "topic": "Remote Work"
    }
    
    result = group_claims(mock_state)
    grouped = result["grouped_data"]
    
    assert len(grouped) == 1
    assert "source1.com" in grouped[0]["supporting_sources"]
    assert "source2.com" in grouped[0]["supporting_sources"]
    
def test_conflicting_claims():
    """Test 3: Preservation of conflicting claims."""
    mock_state = {
        "extracted_claims": {
            "sourceA.com": {"claims": [{"claim": "Remote work increases productivity.", "quote": "Productivity is up."}]},
            "sourceB.com": {"claims": [{"claim": "Remote work destroys productivity.", "quote": "Productivity is down."}]}
        },
        "topic": "Remote Work"
    }
    
    result = group_claims(mock_state)
    grouped = result["grouped_data"]
    
    assert grouped[0]["conflicting_viewpoints"] is not None
    assert len(grouped[0]["conflicting_viewpoints"]) > 5