"""İskelet deneyi: dikkat yerine sıfırdan yeni karıştırıcı (delta | conv), FFN/gömme/normlar donuk."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall.donustur import ConvLM, TeacherConfig  # noqa: E402

HF = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
          vocab_size=97, rope_theta=10000.0, tie_word_embeddings=True)


def _teacher():
    torch.manual_seed(0)
    m = ConvLM(TeacherConfig(HF))
    with torch.no_grad():
        for q in m.parameters():
            q.normal_(0, 0.1) if q.dim() > 1 else None
    for q in m.parameters():
        q.requires_grad_(False)
    return m.eval()


def _skeleton(kind, keep=()):
    m = _teacher()
    ids = torch.randint(0, 97, (2, 40))
    m.set_mode("teacher")
    with torch.no_grad():
        ref = m(ids)
    m.convert_fresh(kind, set(keep))
    for q in m.new_params():
        q.requires_grad_(True)
    return m, ids, ref


@pytest.mark.parametrize("kind", ["delta", "conv"])
def test_teacher_mode_unchanged_by_fresh_mixers(kind):
    m, ids, ref = _skeleton(kind, keep={3})
    m.set_mode("teacher")
    with torch.no_grad():
        assert torch.equal(m(ids), ref)                  # öğretmen yolu bozulmadı


@pytest.mark.parametrize("kind", ["delta", "conv"])
def test_layer_mode_follows_teacher_and_trains_only_new(kind):
    m, ids, ref = _skeleton(kind, keep={3})
    m.set_mode("layer")
    out = m(ids)
    assert torch.allclose(out, ref, atol=1e-5)           # artık akış öğretmeni izler (aşama A)
    loss = m.aux_loss()
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    fresh = [l.self_attn.fresh for i, l in enumerate(m.layers) if i != 3]
    assert all(any(q.grad is not None and q.grad.abs().sum() > 0 for q in f.parameters()) for f in fresh)
    assert m.layers[0].self_attn.q_proj.weight.grad is None and m.layers[0].gate_proj.weight.grad is None
    assert m.layers[3].self_attn.fresh is None            # korunan katman öğretmen dikkati
    assert len(m.new_params()) == sum(len(list(f.parameters())) for f in fresh)


@pytest.mark.parametrize("kind", ["delta", "conv"])
def test_student_is_causal(kind):
    m, ids, _ = _skeleton(kind)
    m.set_mode("student")
    with torch.no_grad():
        a = m(ids)
        ids2 = ids.clone()
        ids2[:, 25:] = (ids2[:, 25:] + 1) % 97
        b = m(ids2)
    assert torch.allclose(a[:, :25], b[:, :25], atol=1e-5)
    assert not torch.allclose(a[:, 25:], b[:, 25:])


@pytest.mark.parametrize("kind", ["delta", "conv"])
def test_stage_a_training_reduces_error(kind):
    m, ids, _ = _skeleton(kind)
    m.set_mode("layer")
    opt = torch.optim.AdamW(m.new_params(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(60):
        m(ids)
        loss = m.aux_loss()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.8 * first, (first, loss.item())


def test_save_conversion_keeps_fresh_params(tmp_path):
    m, ids, _ = _skeleton("delta", keep={3})
    m.save_conversion(tmp_path / "c.pt", {})
    ck = torch.load(tmp_path / "c.pt", weights_only=False)
    assert [c["i"] for c in ck["conv"]] == [0, 1, 2] and all(c["fresh"] == "delta" for c in ck["conv"])
    assert any(".fresh.qkv.weight" in n for n in ck["params"]) and all(".fresh." in n for n in ck["params"])
    assert not any("gate_proj" in n or "q_proj" in n for n in ck["params"])
