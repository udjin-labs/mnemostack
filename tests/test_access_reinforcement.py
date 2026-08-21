"""The access signal in FreshnessBlend: bounded reinforcement, never a penalty.

This term used to be a decay measured from `last_accessed`, which made being
used strictly punishing — a memory recalled once and left cold fell to 0.56
within a month and floored at 0.1 within four, while a memory nothing had
ever found held a flat 1.0 forever. These tests pin the corrected direction
and, just as importantly, its bounds: the one term a recall's own output
feeds back into is also the one that could compound into permanent
favouritism if it were left unbounded.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import FreshnessBlend, PipelineContext
from mnemostack.recall.pipeline.stages import (
    DEFAULT_ACCESS_BONUS_MAX,
    compute_access_boost,
)


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _result(score: float = 1.0, **payload):
    return RecallResult(id="r1", text="memory", score=score, payload=dict(payload))


# --------------------------------------------------- direction and bounds


def test_use_can_only_help_never_hurt():
    """The correction itself. At every age, from fresh to a year cold, a
    memory that has been used scores at least what an unused one does —
    the old decay had this backwards."""
    unused = compute_access_boost(None)
    assert unused == 1.0
    for days in (0, 1, 7, 30, 90, 180, 365, 3650):
        used = compute_access_boost(_iso_days_ago(days), access_count=1)
        assert used >= unused, days


def test_the_bonus_never_exceeds_its_ceiling():
    """The knob is a ceiling, not an asymptote: whatever the count or
    recency, the multiplier stays inside `1 + max_bonus`. Without this the
    one feedback term in the pipeline is unbounded."""
    hottest = compute_access_boost(_iso_days_ago(0), access_count=10**9)
    assert hottest <= 1.0 + DEFAULT_ACCESS_BONUS_MAX
    # ...and the ceiling is REACHED, not merely approached, so `max_bonus`
    # means the bonus an operator actually gets rather than one they never
    # quite see.
    assert hottest == pytest.approx(1.0 + DEFAULT_ACCESS_BONUS_MAX, abs=1e-9)


def test_the_bonus_decays_back_to_neutral_not_below():
    """Reinforcement fades, it does not invert. A long-cold memory returns
    to exactly the score it would have had with no access signal at all."""
    fresh = compute_access_boost(_iso_days_ago(0), access_count=3)
    monthly = compute_access_boost(_iso_days_ago(30), access_count=3)
    ancient = compute_access_boost(_iso_days_ago(3650), access_count=3)
    assert fresh > monthly > ancient
    assert ancient == pytest.approx(1.0, abs=1e-6)
    assert ancient >= 1.0


def test_count_saturates_so_popularity_cannot_compound():
    """Each further access buys strictly less than the one before, and past
    the clamp buys nothing — the guard against a hot memory bidding its way
    to a permanent top slot on a signal its own retrievals generate."""
    # One stamp for every call: recomputing `now` per call would let the
    # clock, not the counter, decide the differences being compared.
    stamp = _iso_days_ago(0)
    at = {n: compute_access_boost(stamp, access_count=n) for n in (1, 2, 9, 10, 1000)}
    first_access_buys = at[2] - at[1]
    tenth_access_buys = at[10] - at[9]
    assert first_access_buys > 0 and tenth_access_buys > 0, at
    assert first_access_buys > tenth_access_buys, at  # diminishing, not linear
    # ...and past the clamp, nothing at all.
    assert at[1000] == pytest.approx(at[10], abs=1e-9)


# ------------------------------------------------------------ the off switch


def test_a_zero_ceiling_takes_the_access_signal_out_of_ranking():
    """The compatibility answer for a deployment that stamps these keys but
    does not want them steering rank — one knob, not a second model."""
    assert compute_access_boost(_iso_days_ago(0), access_count=10, max_bonus=0.0) == 1.0


# ----------------------------------------------------- neutral / bad input


def test_no_last_accessed_is_exactly_neutral():
    """A deployment that never records accesses must see no ranking change
    whatsoever — this is what makes `--record-access` safe to leave off."""
    assert compute_access_boost(None) == 1.0
    assert compute_access_boost("") == 1.0


def test_an_unparseable_stamp_is_neutral_not_a_bonus():
    """Bad metadata must not be a way to buy rank."""
    assert compute_access_boost("not-a-date") == 1.0
    assert compute_access_boost("2026-13-45T99:99:99") == 1.0


def test_a_count_below_one_still_earns_the_bonus_it_evidences():
    """A payload carrying `last_accessed` without a counter is still
    evidence of use; reading the missing count as zero would deny the bonus
    to exactly the memories this term exists to reward."""
    stamped = _iso_days_ago(0)
    assert compute_access_boost(stamped, access_count=0) > 1.0
    assert compute_access_boost(stamped, access_count=0) == pytest.approx(
        compute_access_boost(stamped, access_count=1), abs=1e-9
    )


def test_a_naive_stamp_is_read_as_utc_not_local():
    """Payloads written by older clients carry naive timestamps; reading one
    as local time would shift the bonus by the operator's UTC offset."""
    naive = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    aware = datetime.now(timezone.utc) - timedelta(days=1)
    assert compute_access_boost(naive.isoformat(), access_count=1) == pytest.approx(
        compute_access_boost(aware.isoformat(), access_count=1), abs=1e-6
    )


def test_a_future_stamp_does_not_exceed_the_ceiling():
    """Clock skew must not become extra bonus."""
    future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    assert compute_access_boost(future, access_count=10) == pytest.approx(
        compute_access_boost(_iso_days_ago(0), access_count=10), abs=1e-6
    )


def test_a_degenerate_half_life_does_not_explode_or_invert():
    """Bad config is clamped, not obeyed."""
    for half_life in (0.0, -30.0):
        boost = compute_access_boost(_iso_days_ago(1), access_count=3, half_life_days=half_life)
        assert 1.0 <= boost <= 1.0 + DEFAULT_ACCESS_BONUS_MAX, half_life
    assert compute_access_boost(_iso_days_ago(1), access_count=3, max_bonus=-1.0) == 1.0


# ------------------------------------------------------------- in the stage


def test_the_stage_multiplies_the_score_and_records_the_boost():
    used = _result(score=1.0, last_accessed=_iso_days_ago(0), access_count=10)
    out = FreshnessBlend(weight=0.0).apply(PipelineContext(query="memory"), [used])
    assert out[0].score == pytest.approx(1.0 + DEFAULT_ACCESS_BONUS_MAX, abs=1e-6)
    assert out[0].payload["access_boost"] == pytest.approx(1.0 + DEFAULT_ACCESS_BONUS_MAX, abs=1e-3)


def test_the_stage_leaves_an_unused_memory_exactly_alone():
    """The upgrade contract: a collection with no access keys ranks after
    this change exactly as it did before it."""
    unused = _result(score=1.0)
    out = FreshnessBlend(weight=0.0).apply(PipelineContext(query="memory"), [unused])
    assert out[0].score == pytest.approx(1.0, abs=1e-9)
    assert out[0].payload["access_boost"] == 1.0


def test_the_stage_ceiling_is_configurable_and_zero_disables_it():
    used = _result(score=1.0, last_accessed=_iso_days_ago(0), access_count=10)
    out = FreshnessBlend(weight=0.0, access_bonus_max=0.0).apply(
        PipelineContext(query="memory"), [used]
    )
    assert out[0].score == pytest.approx(1.0, abs=1e-9)


def test_a_corrupt_access_count_in_the_payload_is_not_a_crash():
    """`access_count` arrives from a payload, so it can be anything."""
    for bad in ("many", None, [3]):
        used = _result(score=1.0, last_accessed=_iso_days_ago(0), access_count=bad)
        out = FreshnessBlend(weight=0.0).apply(PipelineContext(query="memory"), [used])
        assert 1.0 <= out[0].score <= 1.0 + DEFAULT_ACCESS_BONUS_MAX, bad


def test_the_bonus_rises_monotonically_across_the_whole_count_range():
    """Reviewer gate: monotone from the first access to the clamp, with no
    plateau or dip in between that a lucky spot-check would miss."""
    stamp = _iso_days_ago(0)
    curve = [compute_access_boost(stamp, access_count=n) for n in range(1, 12)]
    rising = curve[:10]
    assert all(b > a for a, b in zip(rising, rising[1:], strict=False)), curve
    assert curve[-1] == pytest.approx(curve[9], abs=1e-9)  # clamped past 10


def test_the_bonus_falls_monotonically_as_the_access_ages():
    """...and monotone in the other axis too: no age at which waiting
    longer is worth more."""
    curve = [
        compute_access_boost(_iso_days_ago(d), access_count=5)
        for d in (0, 1, 3, 7, 14, 30, 60, 120, 365)
    ]
    assert all(b < a for a, b in zip(curve, curve[1:], strict=False)), curve
    assert curve[-1] >= 1.0


def test_a_negative_access_count_is_not_a_penalty():
    """Reviewer gate: `access_count` comes from a payload, so it can be
    negative. Corrupt data must not become a way to push a memory DOWN —
    the whole point of this term is that it has no downward direction."""
    stamp = _iso_days_ago(0)
    for bad in (-1, -(10**9)):
        boost = compute_access_boost(stamp, access_count=bad)
        assert boost >= 1.0, bad
        # A stamp is evidence of use whatever the counter says, so it reads
        # as the one access it proves — not as zero, and never as a debt.
        assert boost == pytest.approx(compute_access_boost(stamp, access_count=1), abs=1e-9), bad


def test_a_zero_ceiling_is_exactly_the_no_access_signal_score():
    """Reviewer gate, stated as the identity it has to be: with the ceiling
    at 0 the multiplier is literally 1.0, so the score is bit-for-bit what
    it would be with no access keys at all — not merely close to it."""
    stamp = _iso_days_ago(0)
    off = compute_access_boost(stamp, access_count=10, max_bonus=0.0)
    assert off == 1.0

    scored = 0.123456789
    used = _result(score=scored, last_accessed=stamp, access_count=10)
    unused = _result(score=scored)
    ctx = PipelineContext(query="memory")
    with_signal = FreshnessBlend(weight=0.0, access_bonus_max=0.0).apply(ctx, [used])
    without = FreshnessBlend(weight=0.0).apply(ctx, [unused])
    assert with_signal[0].score == without[0].score


def test_without_a_stamp_no_counter_can_buy_a_bonus():
    """The paired opposite of the rule above, and the half that has teeth.

    A stamp is what proves use, and the counter only shapes how much the
    proof is worth — so the gate has to be the stamp alone. Reading the
    counter first, or falling back to it when the stamp is missing, would
    let a payload claiming a thousand accesses and no access time collect
    the ceiling: the counter is caller-supplied metadata, and this term is
    the one a recall's own output feeds back into.
    """
    for count in (0, 1, 10, 10**9, -5):
        for stamp in (None, "", "not-a-date", "2026-13-45T99:99:99"):
            assert compute_access_boost(stamp, access_count=count) == 1.0, (stamp, count)


# ------------------------------------------------- the operator-facing knob


def test_the_off_switch_reaches_an_operator_who_only_has_the_server():
    """The compatibility promise is only true if it is REACHABLE. The knob
    was defensible as an argument against keeping the old model behind a
    legacy flag precisely because it turns the signal off — but a knob only
    a library caller can set is no answer for the deployment that needs it:
    a server whose CLIENTS stamp `last_accessed` cannot escape the term by
    leaving `--record-access` off, because the stage reads those keys
    whoever wrote them.
    """
    import inspect

    from mnemostack.recall.pipeline import build_full_pipeline, build_stateless_pipeline

    for builder in (build_full_pipeline, build_stateless_pipeline):
        assert "access_bonus_max" in inspect.signature(builder).parameters, builder

    pytest.importorskip("fastapi")
    from mnemostack.server import ServerConfig

    assert ServerConfig().access_bonus_max == DEFAULT_ACCESS_BONUS_MAX
    assert ServerConfig(access_bonus_max=0.0).access_bonus_max == 0.0

    parser = pytest.importorskip("mnemostack.cli").build_parser()
    assert parser.parse_args(["serve", "--access-bonus-max", "0"]).access_bonus_max == 0.0


def test_the_configured_ceiling_is_bounded_on_every_construction_path():
    """A knob fed by the service's own output cannot be left to whatever a
    config file says. The bound lives in `__post_init__` so the library
    caller, the env reader and the flag all inherit it — a rule applied at
    each entry point instead would be three rules, and the one path that
    forgot would be the one that mattered."""
    pytest.importorskip("fastapi")
    from mnemostack.recall.pipeline import MAX_ACCESS_BONUS_MAX
    from mnemostack.server import ServerConfig

    # A negative reads as "off", never as an inverted penalty.
    assert ServerConfig(access_bonus_max=-5.0).access_bonus_max == 0.0
    assert ServerConfig(access_bonus_max=99.0).access_bonus_max == MAX_ACCESS_BONUS_MAX
    # NaN compares false against every bound, so a naive clamp lets it
    # through — and it would erase the score of every result it multiplied.
    # It reads as OFF rather than as the default: a ceiling nobody can
    # interpret must not be resolved into a ranking effect nobody asked for.
    assert ServerConfig(access_bonus_max=float("nan")).access_bonus_max == 0.0
    assert ServerConfig(access_bonus_max=float("inf")).access_bonus_max == (MAX_ACCESS_BONUS_MAX)


def test_a_configured_ceiling_actually_reaches_the_stage():
    """Threading a knob through a signature is not the same as it arriving:
    a preset that accepted the argument and dropped it would satisfy every
    check above and change nothing about ranking."""
    from mnemostack.recall.pipeline import FreshnessBlend, build_stateless_pipeline

    off = build_stateless_pipeline(access_bonus_max=0.0)
    stage = next(s for s in off.stages if isinstance(s, FreshnessBlend))
    assert stage.access_bonus_max == 0.0

    used = _result(score=1.0, last_accessed=_iso_days_ago(0), access_count=10)
    out = stage.apply(PipelineContext(query="memory"), [used])
    assert out[0].payload["access_boost"] == 1.0
