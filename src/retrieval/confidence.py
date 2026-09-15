def compute_confidence(scores: list[float]) -> tuple[str, float]:
    """
    Compute confidence from top RRF score.

    RRF scores are small (max ~0.033).
    Thresholds calibrated so High/Medium fire at realistic retrieval ranks.
    """
    if not scores:
        return "Low", 0.0

    top_score = max(scores)

    # Thresholds calibrated for RRF scores (max ~0.033).
    # High   >= 0.020 — ranked highly by both retrievers
    # Medium >= 0.010 — ranked well by at least one retriever
    if top_score >= 0.020:    # Top rank in both searches
        return "High", top_score
    elif top_score >= 0.010:  # Top rank in at least one search
        return "Medium", top_score
    else:
        return "Low", top_score