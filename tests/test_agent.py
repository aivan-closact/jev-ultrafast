"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    a.files = []
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual)["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual)["passed"]


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def upload_page(multiple=False):
    p = page()
    p["actions"].insert(3, {
        "id": "e4", "kind": "upload", "label": "Résumé", "role": "file", "value": "", "node": 40,
        "accept": ".pdf", "multiple": multiple,
    })
    p["fingerprint"] = fingerprint(p)
    return p


def test_file_inputs_are_offered_only_with_caller_supplied_files(tmp_path):
    assert "UPLOAD_FILE" not in model.action_space(upload_page()["actions"])[1]
    assert all(e["role"] != "file" for e in model.action_space(upload_page()["actions"])[0])
    files = [tmp_path / "cv.pdf", tmp_path / "cover.pdf"]
    elements, targets, _ = model.action_space(upload_page()["actions"], files)
    assert elements[2] == {
        "index": "3", "label": "Résumé", "role": "file", "value": "", "accept": ".pdf", "multiple": False,
        "operations": ["UPLOAD_FILE"],
    }
    assert set(targets["UPLOAD_FILE"]) == {"3:1", "3:2"}
    assert targets["UPLOAD_FILE"]["3:2"]["files"] == [str(files[1])]
    assert targets["UPLOAD_FILE"]["3:2"]["id"] == "e4:2"
    assert targets["UPLOAD_FILE"]["3:2"]["label"] == "Résumé ← cover.pdf"
    _, targets, _ = model.action_space(upload_page(multiple=True)["actions"], files)
    assert targets["UPLOAD_FILE"]["3:all"]["files"] == [str(f) for f in files]
    assert model.resolve_action(upload_page(), "e4:1", files)["files"] == [str(files[0])]


def test_model_sees_file_names_and_can_only_pick_an_offered_file(monkeypatch, tmp_path):
    files = [tmp_path / "secret-dir" / "cv.pdf"]
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "UPLOAD_FILE"),
                "upload_file_target": choice(["3:1"], "3:1"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2"], "1"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(upload_page(), "Upload the CV", [], files)
    assert d["choice"] == "e4:1" and d["operation"] == "UPLOAD_FILE" and d["target"] == "3:1"
    assert calls[0]["state"]["files"] == ["cv.pdf"]
    assert calls[0]["questions"]["upload_file_target"]["criteria"]["3:1"]["accept"] == ".pdf"
    assert "secret-dir" not in json.dumps(calls[0])

    def invented(_url, _key, body):
        return {"model": "test", "answers": {
            "operation": choice(body["questions"]["operation"]["criteria"], "UPLOAD_FILE"),
            "upload_file_target": choice(["3:1", "3:2"], "3:2"),
        }}

    monkeypatch.setattr(model, "post_json", invented)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(upload_page(), "Upload the CV", [], files)


def test_agent_rejects_missing_files_before_any_browser_work(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "Browser", Mock(side_effect=AssertionError("must not connect")))
    with pytest.raises(ValueError, match="Not a readable file"):
        loop.Agent("https://example.test/", "Upload", files=[tmp_path / "absent.pdf"])


def test_upload_decision_executes_the_caller_file_on_the_observed_input(runner, tmp_path):
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF")
    runner.files = [cv]
    runner.state["page"] = upload_page()
    runner.state["browser"].observe.return_value = runner.state["page"]
    runner.state["decision"] = {**decision("e4:1"), "operation": "UPLOAD_FILE", "target": "3:1"}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    action = runner.state["browser"].act.call_args.args[0]
    assert action["kind"] == "upload" and action["node"] == 40 and action["files"] == [str(cv)]
    assert runner.state["history"][-1]["action"] == "Résumé ← cv.pdf"
    assert runner.state["history"][-1]["text"] is None


def test_upload_attaches_through_cdp_without_clicking(monkeypatch):
    import jev_ultrafast.browser as browser

    responses = {"Runtime.evaluate": {"result": {"objectId": "handle-1"}}}
    cdp = Mock(side_effect=lambda method, **_: responses.get(method, {}))
    monkeypatch.setattr(browser, "cdp", cdp)
    browser_operation({"operation": "act", "session": "test", "action": {
        "id": "e4:1", "kind": "upload", "node": 40, "files": ["/tmp/cv.pdf"],
    }})
    methods = [c.args[0] for c in cdp.call_args_list]
    assert methods == ["Runtime.evaluate", "DOM.setFileInputFiles", "Runtime.releaseObject"]
    assert cdp.call_args_list[1].kwargs == {"session_id": "test", "files": ["/tmp/cv.pdf"], "objectId": "handle-1"}


def test_upload_without_a_live_input_is_stale_and_a_failed_attach_is_not_retried(monkeypatch):
    import jev_ultrafast.browser as browser

    monkeypatch.setattr(browser, "cdp", Mock(return_value={"result": {"subtype": "null"}}))
    with pytest.raises(StalePage):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e4:1", "kind": "upload", "node": 40, "files": ["/tmp/cv.pdf"],
        }})

    def attach(method, **_):
        if method == "DOM.setFileInputFiles":
            raise RuntimeError("Target closed")
        return {"result": {"objectId": "handle-1"}}

    monkeypatch.setattr(browser, "cdp", Mock(side_effect=attach))
    with pytest.raises(RuntimeError, match="Upload was not confirmed"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e4:1", "kind": "upload", "node": 40, "files": ["/tmp/cv.pdf"],
        }})


def test_upload_cannot_run_without_caller_files():
    with pytest.raises(ValueError):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e4:1", "kind": "upload", "node": 40, "files": [],
        }})
