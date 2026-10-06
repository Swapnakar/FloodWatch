import pytest

from services.safe_route_service import pick_recommendation, risk_level_for_route, route_score


def test_route_score_weighted_composite():
    # max=0.9, p90≈0.88, mean=0.5 -> 0.30*0.9+0.35*0.88+0.35*0.5
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    s = route_score(probs)
    assert s is not None and 0 <= s <= 1
    assert s > max(probs) * 0.3  # dominated by high segments, as intended


def test_route_score_none_when_no_probabilities():
    assert route_score([None, None]) is None
    assert route_score([]) is None


def test_route_score_ignores_none_segments():
    assert route_score([None, 0.4, None, 0.6]) is not None


@pytest.mark.parametrize("score,level", [
    (None, None), (0.8, "HIGH"), (0.5, "ELEVATED"), (0.3, "MODERATE"), (0.1, "LOW"),
])
def test_route_risk_level(score, level):
    assert risk_level_for_route(score) == level


def test_recommends_lower_risk_route_without_saying_safe():
    routes = [{"route_score": 0.2}, {"route_score": 0.6}]
    d = pick_recommendation(routes)
    assert d["recommended_index"] == 0
    assert d["all_risky"] is False
    assert "lower flood risk" in d["recommendation"]
    assert "safe" not in d["recommendation"].lower()


def test_all_routes_risky_warns_explicitly():
    routes = [{"route_score": 0.7}, {"route_score": 0.8}]
    d = pick_recommendation(routes)
    assert d["all_risky"] is True
    assert d["recommended_index"] == 0  # still names the least-bad
    assert "elevated flood risk" in d["recommendation"].lower()
    assert "safe" not in d["recommendation"].lower()


def test_single_route_is_relative_not_guarantee():
    d = pick_recommendation([{"route_score": 0.3}])
    assert d["recommended_index"] == 0
    assert "not a safety guarantee" in d["recommendation"].lower()


def test_no_scores_cannot_assess():
    d = pick_recommendation([{"route_score": None}, {"route_score": None}])
    assert d["recommended_index"] is None
    assert "unable to assess" in d["recommendation"].lower()


def test_similar_routes_reported_as_similar():
    d = pick_recommendation([{"route_score": 0.30}, {"route_score": 0.30}])
    assert "similar" in d["recommendation"].lower()


def test_recommendation_never_claims_safe_across_cases():
    for routes in ([{"route_score": 0.05}], [{"route_score": 0.9}, {"route_score": 0.95}],
                   [{"route_score": 0.1}, {"route_score": 0.8}]):
        text = pick_recommendation(routes)["recommendation"].lower()
        # "safety guarantee" (honest disclaimer) is fine; "is safe" / "100% safe" is not.
        assert "100% safe" not in text
        assert "is safe" not in text
        assert "route is safe" not in text
