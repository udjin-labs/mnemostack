"""Deterministic synthetic retrieval fixtures; no network or model."""

import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 5, 15, 12, tzinfo=timezone.utc)


def generate():
    rng = random.Random(232)
    memories, queries = [], []

    def note(text, age=30, source="journal.md", **extra):
        mid = len(memories) + 1
        memories.append(
            dict(
                id=mid,
                text=text,
                timestamp=(NOW - timedelta(days=age)).isoformat(),
                source=source,
                **extra,
            )
        )
        return mid

    pairs = [
        (
            "How can I silence the ringing handset?",
            "Mute the telephone bell using the side switch.",
        ),
        ("Where does the physician reside?", "The doctor lives beside the harbor."),
        ("Как добраться до больницы?", "Take the tram toward the medical center."),
        ("Which meal does the child prefer?", "Ребёнок обожает гречневую кашу."),
        ("Where may automobiles remain overnight?", "Vehicles can park inside the garage."),
        ("Как согреть комнату?", "Turn up the radiator thermostat."),
        ("How should I mend damaged trousers?", "Заштопай порванные брюки иголкой."),
        ("How can I purchase inexpensive footwear?", "Buy cheap shoes at the outlet."),
    ]
    for kind in ("exact_code", "date", "old_vs_recent", "paraphrase", "fallback", "plain"):
        for i in range(40):
            entity = f"{['Cedar', 'Birch', 'Maple', 'Willow', 'Elm'][i % 5]} {i // 5 + 1}"
            ru = i % 5 == 0
            age, source, extra = 40, "journal.md", {}
            if kind == "exact_code":
                code = [
                    str(8100 + i),
                    f"10.23.{i}.7",
                    f"2026.4.{i + 1}",
                    f"ABC-{1200 + i}",
                    f"ERR-{4200 + i}",
                ][i % 5]
                field = ["port", "IP address", "version", "ticket id", "error code"][i % 5]
                q = (
                    f"What is the production {field} for {entity}?"
                    if not ru
                    else f"Какой {field} у сервиса {entity}?"
                )
                text = f"{entity} production {field}: {code}. Approved active setting."
                source = ["MEMORY.md", "TOOLS.md"][i % 2]
                wrong = [
                    f"{entity} staging {field}: 9999. Testing only, never production.",
                    f"{entity} retired production {field}: 7001. Superseded.",
                    f"Bookkeeping: {code} receipts counted; unrelated to infrastructure.",
                    f"{entity} proposed {field}: 9090. Proposal rejected.",
                ]
            elif kind == "date":
                dates = ["2025-04-30", "2025-04-30", "2026-04-08", "2024-08-10"]
                d = datetime.fromisoformat(dates[i % 4]).replace(tzinfo=timezone.utc)
                age = (NOW - d).total_seconds() / 86400
                period = ["2025-04-30", "April 30, 2025", "last month", "2024"][i % 4]
                q = f"What did the {entity} committee decide in {period}?"
                text = f"{entity} committee approved the riverside venue." + (
                    f" Meeting date: {dates[i % 4]}."
                    if i % 4 != 3
                    else " Minutes signed after the summer session."
                )
                wrong = [
                    f"{entity} committee postponed the riverside venue discussion. Meeting date: 2026-05-14.",
                    f"{entity} committee rejected the hilltop venue. Meeting date: 2023-04-30.",
                ]
            elif kind == "old_vs_recent":
                age = 180 + i * 13
                q = (
                    f"Which cabinet holds the original signed {entity} lease?"
                    if not ru
                    else f"Где хранится оригинал договора {entity}?"
                )
                text = f"Original signed {entity} lease is in cabinet violet, drawer seven."
                wrong = [
                    f"{entity} lease team discussed new stationery.",
                    f"{entity} office ordered a cabinet catalog.",
                    f"{entity} lease reminder: schedule next meeting.",
                    f"{entity} staff checked drawer labels.",
                ]
            elif kind == "paraphrase":
                q, text = pairs[i % len(pairs)]
                # Different contexts identify the intended episode without lexical anchors.
                q = (
                    q
                    + " "
                    + [
                        "During the winter visit.",
                        "After the summer trip.",
                        "On the spring holiday.",
                        "Following the autumn journey.",
                        "At the weekend retreat.",
                    ][i // 8]
                )
                text += (
                    " "
                    + [
                        "Cold season guest instructions.",
                        "Hot weather travel memo.",
                        "Blossom vacation advice.",
                        "Harvest excursion briefing.",
                        "Saturday getaway guidance.",
                    ][i // 8]
                )
                wrong = [
                    q + " No answer was established; this was merely an agenda item.",
                    q + " Earlier suggestion withdrawn as incorrect.",
                ]
            elif kind == "fallback":
                q = f"Find the sealed recovery phrase for vault {entity}."
                text = f"Sealed recovery phrase for vault {entity}: amber heron sails quietly."
                extra = {"fallback_only": True}
                wrong = [
                    f"Vault {entity} recovery phrase review scheduled; phrase is absent from this agenda."
                ]
            else:
                q = (
                    f"What tea does {entity} prefer at breakfast?"
                    if not ru
                    else f"Какой чай любит {entity} за завтраком?"
                )
                text = (
                    f"{entity} prefers jasmine tea at breakfast."
                    if not ru
                    else f"{entity} любит жасминовый чай за завтраком."
                )
                wrong = [
                    f"{entity} prefers coffee after dinner.",
                    f"{entity} bought mint tea as a gift, not for breakfast.",
                ]
            gold = note(text, age, source, **extra)
            queries.append(
                dict(
                    id=f"{kind}-{i:02}",
                    kind=kind,
                    text=q,
                    gold=[gold],
                    fault="empty_primary_vector" if kind == "fallback" else None,
                )
            )
            for j, w in enumerate(wrong):
                note(
                    w,
                    [2 / 1440, 1 / 24, 1, 3][j % 4] if kind == "old_vs_recent" else 3 + j * 90,
                    ["MEMORY.md", "TOOLS.md", "meeting.md", "archive.md"][j % 4],
                )
    while len(memories) < 3000:
        i = len(memories)
        text = (
            f"Заметка {i}: на встрече отдела обсудили отпуск и заказ обедов."
            if i % 5 == 0
            else f"Office memo {i}: team {rng.randrange(100)} reviewed lunch orders and holiday coverage."
        )
        note(text, rng.randrange(1, 900))
    return dict(
        schema=1, reference_now=NOW.isoformat(), seed=232, memories=memories, queries=queries
    )


if __name__ == "__main__":
    data = generate()
    with open("corpus.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("memories:", len(data["memories"]))
    print(dict(Counter(q["kind"] for q in data["queries"])))
