import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_llm: marks tests that require LLM model loading"
    )
    config.addinivalue_line(
        "markers", "requires_prover: marks tests that require Rust prover binary"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names."""
    for item in items:
        # Mark E2E tests as slow and requiring LLM
        if "E2EBasicFlow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_llm)
        
        # Mark proof tests as requiring prover
        if "prove" in item.name.lower():
            item.add_marker(pytest.mark.requires_prover)

