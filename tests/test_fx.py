from utils import fx


def test_minor_unit_fx_rate_scales_when_canonical_currency_matches():
    assert fx.fx_rate_to_base("GBP", "GBp")["rate"] == 100.0
    assert fx.fx_rate_to_base("GBP", "GBX")["rate"] == 100.0
    assert fx.fx_rate_to_base("GBp", "GBP")["rate"] == 0.01
    assert fx.fx_rate_to_base("GBp", "GBX")["rate"] == 1.0


def test_minor_unit_fx_rate_scales_cross_currency_target(monkeypatch):
    monkeypatch.setattr(fx, "_fetch_fx_rate_uncached", lambda currency, base: {"rate": 2.0, "as_of": "2026-05-08"})

    quote = fx.fx_rate_to_base("USD", "GBp")

    assert quote == {"rate": 200.0, "as_of": "2026-05-08"}
