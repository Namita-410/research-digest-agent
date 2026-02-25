import pytest
from graph_agent import ingest_sources, group_claims, AgentState

def test_empty_unreachable_source():
    """Test 1: Empty/unreachable source handling safely."""
    # We mock a state with a bad URL
    mock_state = {"urls": ["https://this-website-does-not-exist-12345.com"]}
    
    result = ingest_sources(mock_state)
    scraped = result["scraped_data"][0]
    
    assert scraped["status"] == "failed"
    assert "error" in scraped
    assert scraped["content"] == ""

def test_deduplication():
    """Test 2: Deduplication of duplicate content."""
    # Mocking two sources saying the exact same thing
    mock_state = {
        "extracted_claims": {
            "source1.com": {"claims": [{"claim": "AI is growing fast.", "quote": "AI grows fast."}]},
            "source2.com": {"claims": [{"claim": "Artificial Intelligence is expanding rapidly.", "quote": "AI expands rapidly."}]}
        }
    }
    
    result = group_claims(mock_state)
    grouped = result["grouped_data"]
    
    # The LLM should group these identical concepts into exactly 1 theme
    assert len(grouped) == 1
    assert "source1.com" in grouped[0]["supporting_sources"]
    assert "source2.com" in grouped[0]["supporting_sources"]

def test_conflicting_claims():
    """Test 3: Preservation of conflicting claims."""
    # Mocking two sources completely disagreeing
    mock_state = {
        "extracted_claims": {
            "sourceA.com": {"claims": [{"claim": "Remote work increases productivity.", "quote": "Productivity is up."}]},
            "sourceB.com": {"claims": [{"claim": "Remote work destroys productivity.", "quote": "Productivity is down."}]}
        }
    }
    
    result = group_claims(mock_state)
    grouped = result["grouped_data"]
    
    # The LLM should capture the conflict in the conflicting_viewpoints field
    assert grouped[0]["conflicting_viewpoints"] is not None
    assert len(grouped[0]["conflicting_viewpoints"]) > 5 # Ensure it actually wrote an explanation