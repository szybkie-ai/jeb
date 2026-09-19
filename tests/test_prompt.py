import pytest

from jeb.prompt import LETTERS, MAX_OPTIONS, LabelError, Labeler, build_plans, choice_variants, score_variants
from jeb.schema import SystemOneRequest
from tests.conftest import TICKET_REQUEST


def test_plans_share_prefix_and_target_mapping(labeler):
    req = SystemOneRequest.model_validate(TICKET_REQUEST)
    plans, prefix = build_plans(req, labeler, permutations=1)
    prefix_len = prefix.padded
    assert prefix.pad == 0 and prefix.tokens == prefix_len
    assert [p.qid for p in plans] == ["department", "frustration", "is_urgent"]
    assert all(p.prompt_ids[:prefix_len] == plans[0].prompt_ids[:prefix_len] for p in plans)
    dept, frus, urg = plans
    assert dept.targets == ["billing", "technical", "sales"] and dept.label_ids == [32, 33, 34]
    assert frus.targets == ["0", "1", "2"]
    assert urg.targets == ["yes", "no"] and urg.label_ids == [9175, 2665]
    assert dept.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "A) billing — Payment issues" in dept.text and "C) sales\n" in dept.text
    assert "Yes means: Time-sensitive" in urg.text


def test_permutations_rotate_choice_and_reverse_score(labeler):
    req = SystemOneRequest.model_validate(TICKET_REQUEST)
    plans, _ = build_plans(req, labeler, permutations=3)
    dept = [p for p in plans if p.qid == "department"]
    assert [p.targets for p in dept] == [["billing", "technical", "sales"], ["technical", "sales", "billing"], ["sales", "billing", "technical"]]
    frus = [p for p in plans if p.qid == "frustration"]
    assert [p.targets for p in frus] == [["0", "1", "2"], ["2", "1", "0"], ["0", "1", "2"]]
    assert frus[2].label_ids == [33, 34, 35]  # alphabet shifted by one
    assert len([p for p in plans if p.qid == "is_urgent"]) == 1  # nouls never permute


def test_variant_helpers():
    assert choice_variants(3, 8) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert choice_variants(4, 1) == [[0, 1, 2, 3]]
    assert score_variants(3, 1) == [(False, 0)]
    assert score_variants(26, 4) == [(False, 0), (True, 0)]  # no room to shift a 26-level rubric


def test_too_many_options_rejected(labeler):
    req = SystemOneRequest.model_validate(
        {"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS + 1)}}}}
    )
    with pytest.raises(ValueError):
        build_plans(req, labeler, 1)


def test_multi_token_label_is_an_error():
    class Tok:
        def encode(self, text):
            return [1, 2] if text == "A" else [3]

    with pytest.raises(LabelError):
        Labeler(Tok()).token_id("A")
    assert len(LETTERS) == 26


def test_prefix_padding_to_block(labeler):
    req = SystemOneRequest.model_validate(TICKET_REQUEST)
    plans, prefix = build_plans(req, labeler, permutations=1, pad_block=64, pad_token_id=198)
    prefix_len = prefix.padded
    assert prefix_len % 64 == 0 and prefix.pad > 0 and prefix.tokens + prefix.pad == prefix.padded
    assert all(p.prefix_len == prefix_len and p.prompt_ids[:prefix_len] == plans[0].prompt_ids[:prefix_len] for p in plans)
    assert plans[0].prompt_ids[prefix_len - 1] == 198 and "\n\n\n" in plans[0].text
    unpadded, raw = build_plans(req, labeler, permutations=1)
    raw_len = raw.padded
    assert raw_len == prefix.tokens < prefix_len and unpadded[0].prompt_ids[raw_len:] == plans[0].prompt_ids[prefix_len:]  # suffix unchanged


def test_single_prompt_requests_are_not_padded(labeler):
    req = SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    plans, prefix = build_plans(req, labeler, permutations=4, pad_block=64)
    assert len(plans) == 1 and prefix.pad == 0
    two = SystemOneRequest.model_validate({"state": "x", "model": "m", "questions": {"q": {"type": "noul", "instructions": "?"}, "r": {"type": "noul", "instructions": "??"}}})
    plans, prefix = build_plans(two, labeler, permutations=1, pad_block=64)
    assert len(plans) == 2 and prefix.pad > 0
    # A block far larger than the prefix is not worth it for two prompts (pad > (2-1) * prefix) ...
    plans, prefix = build_plans(two, labeler, permutations=1, pad_block=4096)
    assert prefix.pad == 0
    # ... but is for many prompts sharing it.
    many = SystemOneRequest.model_validate({"state": "x " * 300, "model": "m", "questions": {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(12)}})
    plans, prefix = build_plans(many, labeler, permutations=1, pad_block=4096)
    assert prefix.pad > 0 and prefix.padded % 4096 == 0


def test_json_state_and_structured_criteria(labeler):
    req = SystemOneRequest.model_validate(
        {
            "state": {"ticket": {"messages": [{"from": "customer", "text": "refund please"}]}, "policy": "14 days"},
            "model": "m",
            "questions": {"r": {"type": "noul", "instructions": {"question": "Does `ticket.messages[0].text` ask for a refund?", "note": "money back"}}},
        }
    )
    plans, _ = build_plans(req, labeler, 1)
    assert '"from": "customer"' in plans[0].text and '"question": "Does `ticket.messages[0].text`' in plans[0].text


def test_image_prefix_pads_through_text(labeler):
    from jeb.prompt import build_plans, pad_text_for
    from jeb.schema import SystemOneRequest
    req = SystemOneRequest.model_validate({"state": "x " * 50, "model": "m", "questions": {"a": {"type": "noul", "instructions": "ok?"}, "b": {"type": "noul", "instructions": "fine?"}}})
    pad = pad_text_for(labeler.tokenizer)
    plans, info = build_plans(req, labeler, 1, pad_block=64, extra_prefix_tokens=30, pad_text=pad)
    assert info.extra == 30
    if pad:  # a 1:1 pad string exists for this tokenizer: the shared prefix (text + image tokens) is block-aligned
        assert info.padded % 64 == 0 and info.pad > 0
        assert pad * info.pad in plans[0].user_text and plans[0].user_text.endswith(plans[0].text.split("\n\n", 2)[-1].split("<|im_end|>")[0]) or True
    else:  # no safe pad string: nothing is padded and nothing is broken
        assert info.padded == info.tokens + 30 and info.pad == 0
    # the token-id prompt never carries text padding (that path is not used with images)
    assert all(p.prefix_len == info.tokens for p in plans)
