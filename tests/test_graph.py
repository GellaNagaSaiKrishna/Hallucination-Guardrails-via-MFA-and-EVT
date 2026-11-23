from service.graph import OOD_MESSAGE, PREMISE_MESSAGE, build_graph


def test_ood_question_is_refused_before_retrieval(components):
    out = build_graph(components).invoke({"question": "Which Mario game first introduced Wario?"})
    assert out["refused"] and out["refusal_reason"] == "out_of_distribution"
    assert out["answer"] == OOD_MESSAGE
    assert out["trace"] == ["embed", "gate", "refuse"]
    assert components.store.calls == 0          # retriever never touched


def test_false_premise_is_refused_after_retrieval(components):
    out = build_graph(components).invoke({"question": "Why did Costco's net sales decline in 2021?"})
    assert out["refused"] and out["refusal_reason"] == "unsupported_premise"
    assert out["answer"] == PREMISE_MESSAGE
    assert out["trace"] == ["embed", "gate", "retrieve", "premise_check", "refuse"]
    assert out["premise"]["supported"] is False


def test_answerable_question_is_generated_with_evidence(components):
    out = build_graph(components).invoke({"question": "What drove the increase in Costco's net sales in 2021?"})
    assert not out["refused"] and out["refusal_reason"] is None
    assert out["answer"] == "ANSWER(18%)"
    assert out["trace"] == ["embed", "gate", "retrieve", "premise_check", "generate"]
    assert len(out["hits"]) == 3                 # default top_k from Components


def test_top_k_override_and_timings(components):
    out = build_graph(components).invoke({"question": "Costco net sales 2021?", "top_k": 5})
    assert len(out["hits"]) == 5
    assert set(out["timings_ms"]) == {"embed", "gate", "retrieve", "premise_check", "generate"}
    assert all(v >= 0 for v in out["timings_ms"].values())


def test_noop_checker_passes_everything(components):
    from service.premise import NoopChecker
    components.checker = NoopChecker()
    out = build_graph(components).invoke({"question": "Why did Costco's net sales decline in 2021?"})
    assert not out["refused"]


def test_company_filter_is_passed_to_the_store(components):
    out = build_graph(components).invoke({"question": "What drove the increase in Costco's net sales in 2021?"})
    assert out["companies"] == ["Costco"] and components.store.last_companies == ["Costco"]
    out = build_graph(components).invoke({"question": "What drove the increase in net sales in 2021?"})
    assert out["companies"] == [] and components.store.last_companies is None
    components.company_filter = False
    build_graph(components).invoke({"question": "What drove the increase in Costco's net sales in 2021?"})
    assert components.store.last_companies is None
