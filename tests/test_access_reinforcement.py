"""The access signal in FreshnessBlend: bounded reinforcement, never a penalty.

This term used to be a decay measured from `last_accessed`, which made being
used strictly punishing — a memory recalled once and left cold fell to 0.56
within a month and floored at 0.1 within four, while a memory nothing had
ever found held a flat 1.0 forever. These tests pin the corrected direction
and, just as importantly, its bounds: the one term a recall's own output
feeds back into is also the one that could compound into permanent
favouritism if it were left unbounded.
"""

import os
import time
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
    as local time would shift the bonus by the operator's UTC offset.

    The offset is FORCED rather than inherited from the host. Comparing a
    naive stamp against an aware one proves nothing on a machine that is
    already UTC — the two readings coincide whatever the code does, so on
    a UTC runner (which CI is) this test would pass over the very bug it
    names. TZ is pinned to a zone with a large offset so the difference
    has somewhere to show up.
    """
    if not hasattr(time, "tzset"):
        # Windows has no tzset, so the offset cannot be forced there. Skip
        # rather than fall back to the host zone: a green run that proves
        # nothing is worse than an honest gap, and this project's CI is
        # Linux — but contributors are not.
        pytest.skip("TZ cannot be forced on this platform (no time.tzset)")
    # Restored by hand rather than with monkeypatch: monkeypatch puts the
    # TZ variable back but cannot call `tzset()` afterwards, so libc would
    # stay at UTC+14 for every later test in this process and make the
    # timezone-sensitive ones order-dependent. And "no previous value" is
    # an ABSENCE, not an empty string — an empty TZ means UTC to libc,
    # which is a different setting than not having one.
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "Pacific/Kiritimati"  # UTC+14
    time.tzset()
    try:
        naive = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
        aware = datetime.now(timezone.utc) - timedelta(days=1)
        assert compute_access_boost(naive.isoformat(), access_count=1) == pytest.approx(
            compute_access_boost(aware.isoformat(), access_count=1), abs=1e-6
        )
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


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


def test_the_stage_positional_surface_is_pinned():
    """Third time this class of defect appeared in one change, so it gets a
    pin rather than a fourth act of remembering.

    `access_bonus_max` was first inserted mid-signature, where
    `FreshnessBlend(0.2, 14, None, 60, 0.5)` bound 60 to the bonus ceiling
    and 0.5 to `echo_window_minutes`. This is NOT a stability promise —
    `docs/api-stability.md` files the pipeline stages as experimental and
    tells callers to use `Recaller` / `recall_flow` rather than assemble
    stages themselves. It is pinned because of how the mistake FAILS: 60
    clamps to a legal 1.0 instead of raising, so the recall simply ranks
    differently and nothing anywhere says why.

    The knob is keyword-only now, which is stronger than appending: past
    the star no future parameter can shift another one either.
    """
    import inspect

    from mnemostack.recall.pipeline import FreshnessBlend

    params = inspect.signature(FreshnessBlend.__init__).parameters
    positional = [
        name
        for name, p in params.items()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and name != "self"
    ]
    assert positional == [
        "weight",
        "halflife_days",
        "confidence_half_life_days",
        "echo_window_minutes",
        "echo_penalty",
        "always_current_files",
        "always_current_freshness",
        "timestamp_key",
        "timestamp_format",
    ], positional
    assert params["access_bonus_max"].kind is inspect.Parameter.KEYWORD_ONLY

    # ...and the call that would have broken, spelled out: the fourth
    # positional is the echo window, not a bonus ceiling.
    stage = FreshnessBlend(0.2, 14, None, 60, 0.5)
    assert stage.echo_window_minutes == 60
    assert stage.echo_penalty == 0.5
    assert stage.access_bonus_max == DEFAULT_ACCESS_BONUS_MAX


# --------------------------------------- overflow, and every recall surface


def test_a_payload_counter_cannot_crash_the_recall():
    """P1 from review. `access_count` arrives from a payload, and a remote
    caller can put a thousand-digit number there through ordinary metadata:
    it passes the metadata size limits, becomes a Python int, and `float()`
    on it raises OverflowError — aborting the WHOLE recall. A read path
    must not be crashable by a value somebody else wrote."""
    stamp = _iso_days_ago(0)
    huge = 10**400
    assert compute_access_boost(stamp, access_count=huge) == pytest.approx(
        compute_access_boost(stamp, access_count=10), abs=1e-9
    )
    assert compute_access_boost(stamp, access_count=-huge) >= 1.0

    used = _result(score=1.0, last_accessed=stamp, access_count=huge)
    out = FreshnessBlend(weight=0.0).apply(PipelineContext(query="q"), [used])
    assert 1.0 <= out[0].score <= 1.0 + DEFAULT_ACCESS_BONUS_MAX


def test_an_overflowing_ceiling_clamps_instead_of_raising():
    """Same shape on the configuration side: `float(10**1000)` raises
    before any clamp can run, so a programmatic caller passing it to a
    public builder crashed construction instead of getting the cap."""
    from mnemostack.recall.pipeline import MAX_ACCESS_BONUS_MAX, normalize_access_bonus_max

    assert normalize_access_bonus_max(10**400) == MAX_ACCESS_BONUS_MAX
    assert normalize_access_bonus_max(-(10**400)) == 0.0
    assert normalize_access_bonus_max("nonsense") == 0.0


def test_every_recall_command_honours_the_off_switch():
    """Three separate surfaces shipped without this knob before it was
    caught — the public builders, then MCP, then `search`/`answer`. So the
    check is over the whole SET rather than one more entry point: an off
    switch that depends on which command you typed is not an off switch.
    """
    from mnemostack.cli import build_parser

    parser = build_parser()
    for argv in (
        ["serve", "--access-bonus-max", "0"],
        ["search", "q", "--access-bonus-max", "0"],
        ["answer", "q", "--access-bonus-max", "0"],
        ["mcp-serve", "--access-bonus-max", "0"],
    ):
        assert parser.parse_args(argv).access_bonus_max == 0.0, argv


def test_the_environment_reaches_every_surface_through_one_resolver(monkeypatch):
    """...and the env var likewise. It is read in one place so that the
    answer cannot differ between commands — the failure mode when each
    entry point retypes its own wiring."""
    from mnemostack.recall.pipeline import (
        DEFAULT_ACCESS_BONUS_MAX as DEFAULT,
    )
    from mnemostack.recall.pipeline import (
        MAX_ACCESS_BONUS_MAX,
        resolve_access_bonus_max,
    )

    monkeypatch.delenv("MNEMOSTACK_ACCESS_BONUS_MAX", raising=False)
    assert resolve_access_bonus_max() == DEFAULT

    monkeypatch.setenv("MNEMOSTACK_ACCESS_BONUS_MAX", "0")
    assert resolve_access_bonus_max() == 0.0
    # An explicit setting still beats the environment.
    assert resolve_access_bonus_max(1.0) == MAX_ACCESS_BONUS_MAX

    # A typo must not fail startup, and must not silently mean "off" —
    # it falls back to the default, like every other tuning knob here.
    monkeypatch.setenv("MNEMOSTACK_ACCESS_BONUS_MAX", "not-a-number")
    assert resolve_access_bonus_max() == DEFAULT


def test_the_flag_and_the_variable_never_disagree(monkeypatch):
    """Found by reading a reviewer's own probe output rather than its
    findings: `99` clamped to the cap both ways, but `inf` clamped via the
    flag and fell back to the DEFAULT via the environment — 1.0 against
    0.25 for the same intent, expressed two ways.

    That is the branch's recurring defect wearing a different hat: not "one
    surface is missing the knob" but "two ways of setting it disagree".
    Asking for no ceiling gets the highest one allowed, exactly as 99 does,
    whichever way you ask.
    """
    from mnemostack.recall.pipeline import (
        DEFAULT_ACCESS_BONUS_MAX as DEFAULT,
    )
    from mnemostack.recall.pipeline import (
        resolve_access_bonus_max,
    )

    for text in ("0", "0.4", "99", "1e400", "inf", "-1", "-inf"):
        monkeypatch.setenv("MNEMOSTACK_ACCESS_BONUS_MAX", text)
        from_env = resolve_access_bonus_max()
        from_flag = resolve_access_bonus_max(float(text))
        assert from_env == from_flag, (text, from_env, from_flag)

    # Only a genuinely unparseable value falls back — and to the default,
    # not to "off": a typo must not silently disable a ranking signal.
    monkeypatch.setenv("MNEMOSTACK_ACCESS_BONUS_MAX", "not-a-number")
    assert resolve_access_bonus_max() == DEFAULT
