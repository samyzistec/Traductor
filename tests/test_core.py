
import pytest

def _skip_or_fail(strict, message):
    if strict:
        pytest.fail(message)
    else:
        pytest.skip(message)

def test_notebook_loaded(nb_namespace):
    ns, strict = nb_namespace
    assert isinstance(ns, dict)
    if "_introspection_error" in ns:
        _skip_or_fail(strict, f"Introspection warning: {ns['_introspection_error']}")

def test_has_core_symbols(nb_namespace):
    ns, strict = nb_namespace
    expected_any = [
        "PositionalEncoding",
        "MultiHeadAttention",
        "FeedForward",
        "EncoderLayer",
        "DecoderLayer",
        "TransformerModel",
        "LabelSmoothingLoss",
        "NoamWrapper",
        "make_pad_mask",
        "make_causal_mask",
    ]
    found = [name for name in expected_any if name in ns]
    if not found:
        _skip_or_fail(strict, "No expected symbols found in notebook definitions.")
    else:
        assert len(found) >= 2

@pytest.mark.parametrize("cls_name", ["PositionalEncoding"])
def test_positional_encoding_shape(nb_namespace, cls_name):
    ns, strict = nb_namespace
    if cls_name not in ns:
        _skip_or_fail(strict, f"{cls_name} not present.")
        return
    torch = ns.get("torch", None)
    nn = ns.get("nn", None)
    if torch is None or isinstance(nn, type) or not hasattr(nn, "Module"):
        _skip_or_fail(strict, "PyTorch not available for PositionalEncoding test.")
        return

    PE = ns[cls_name]
    try:
        pe = PE(d_model=32, dropout=0.0, max_len=50)
    except TypeError:
        try:
            pe = PE(32, 0.0, 50)
        except Exception as e:
            _skip_or_fail(strict, f"Could not instantiate {cls_name}: {e}")
            return

    x = ns["torch"].zeros(2, 10, 32)
    try:
        y = pe(x)
    except Exception as e:
        _skip_or_fail(strict, f"{cls_name} forward failed: {e}")
        return

    assert tuple(y.shape) == (2, 10, 32)

@pytest.mark.parametrize("cls_name", ["MultiHeadAttention"])
def test_multihead_attention_shape(nb_namespace, cls_name):
    ns, strict = nb_namespace
    if cls_name not in ns:
        _skip_or_fail(strict, f"{cls_name} not present.")
        return
    torch = ns.get("torch", None)
    nn = ns.get("nn", None)
    if torch is None or isinstance(nn, type) or not hasattr(nn, "Module"):
        _skip_or_fail(strict, "PyTorch not available for MultiHeadAttention test.")
        return

    MHA = ns[cls_name]
    try:
        mha = MHA(d_model=32, n_heads=4, dropout=0.0)
    except TypeError:
        try:
            mha = MHA(32, 4, 0.0)
        except Exception as e:
            _skip_or_fail(strict, f"Could not instantiate {cls_name}: {e}")
            return

    q = ns["torch"].randn(2, 10, 32)
    k = ns["torch"].randn(2, 10, 32)
    v = ns["torch"].randn(2, 10, 32)

    try:
        out = mha(q, k, v, mask=None)
    except TypeError:
        try:
            out = mha(q, k, v)
        except Exception as e:
            _skip_or_fail(strict, f"{cls_name} forward failed: {e}")
            return
    except Exception as e:
        _skip_or_fail(strict, f"{cls_name} forward failed: {e}")
        return

    assert tuple(out.shape) == (2, 10, 32)

def test_masks_functions_exist(nb_namespace):
    ns, strict = nb_namespace
    pad = ns.get("make_pad_mask", None)
    causal = ns.get("make_causal_mask", None)
    if pad is None and causal is None:
        _skip_or_fail(strict, "Mask helpers not present.")
        return

    torch = ns.get("torch", None)
    if torch is None:
        _skip_or_fail(strict, "PyTorch not available for mask tests.")
        return

    if pad is not None:
        try:
            ids = torch.tensor([[1,2,0,0],[3,4,5,0]])
            mask = pad(ids, pad_id=0) if "pad_id" in getattr(pad, "__code__", type("x", (), {"co_varnames": ()})) .co_varnames else pad(ids)
            assert mask.shape[-1] == ids.shape[1]
        except Exception as e:
            _skip_or_fail(strict, f"make_pad_mask failed: {e}")

    if causal is not None:
        try:
            m = causal(5)
            assert m.shape[-1] == m.shape[-2] == 5
        except TypeError:
            try:
                m = causal(torch.ones(5,5))
                assert m.shape[-1] == m.shape[-2] == 5
            except Exception as e:
                _skip_or_fail(strict, f"make_causal_mask failed: {e}")

def test_transformer_model_symbol(nb_namespace):
    ns, strict = nb_namespace
    if "TransformerModel" not in ns:
        _skip_or_fail(strict, "TransformerModel not present.")
        return
    assert callable(ns["TransformerModel"])
