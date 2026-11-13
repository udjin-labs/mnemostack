# BASE versus MERGED

Differences are second minus first. Missing rank is worse than any observed rank. MRR uses the evaluated candidate pool; recall_flow ranks are censored at its output limit.

## raw / raw

| kind | metric | first | second | difference |
|---|---|---:|---:|---:|
| date | n | 40.0000 | 40.0000 | +0.0000 |
| date | MRR | 0.7750 | 0.7750 | +0.0000 |
| date | hit@1 | 0.5500 | 0.5500 | +0.0000 |
| date | hit@3 | 1.0000 | 1.0000 | +0.0000 |
| date | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| date | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| date | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| date | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | n | 40.0000 | 40.0000 | +0.0000 |
| exact_code | MRR | 0.5625 | 0.5625 | +0.0000 |
| exact_code | hit@1 | 0.3000 | 0.3000 | +0.0000 |
| exact_code | hit@3 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| fallback | n | 40.0000 | 40.0000 | +0.0000 |
| fallback | MRR | 0.0476 | 0.0476 | +0.0000 |
| fallback | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | n | 40.0000 | 40.0000 | +0.0000 |
| old_vs_recent | MRR | 0.8938 | 0.8938 | +0.0000 |
| old_vs_recent | hit@1 | 0.8250 | 0.8250 | +0.0000 |
| old_vs_recent | hit@3 | 0.9750 | 0.9750 | +0.0000 |
| old_vs_recent | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| overall | n | 240.0000 | 240.0000 | +0.0000 |
| overall | MRR | 0.5575 | 0.5575 | +0.0000 |
| overall | hit@1 | 0.4458 | 0.4458 | +0.0000 |
| overall | hit@3 | 0.6625 | 0.6625 | +0.0000 |
| overall | hit@5 | 0.6667 | 0.6667 | +0.0000 |
| overall | recall@5 | 0.6667 | 0.6667 | +0.0000 |
| overall | recall@10 | 0.6667 | 0.6667 | +0.0000 |
| overall | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| paraphrase | n | 40.0000 | 40.0000 | +0.0000 |
| paraphrase | MRR | 0.0662 | 0.0662 | +0.0000 |
| paraphrase | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| plain | n | 40.0000 | 40.0000 | +0.0000 |
| plain | MRR | 1.0000 | 1.0000 | +0.0000 |
| plain | hit@1 | 1.0000 | 1.0000 | +0.0000 |
| plain | hit@3 | 1.0000 | 1.0000 | +0.0000 |
| plain | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@25 | 1.0000 | 1.0000 | +0.0000 |

| kind | pairs | better | worse | same | entered 5 | left 5 | entered 10 | left 10 | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 240 | 0 | 0 | 240 | 0 | 0 | 0 | 0 | 1 |
| date | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| exact_code | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| fallback | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| old_vs_recent | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| paraphrase | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| plain | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |

### 30 largest regressions (reciprocal-rank change)


### 30 largest improvements (reciprocal-rank change)


Unpaired/error queries: 0. Recorded skips: first 0, second 0.

## stateless / stateless

| kind | metric | first | second | difference |
|---|---|---:|---:|---:|
| date | n | 40.0000 | 40.0000 | +0.0000 |
| date | MRR | 0.0914 | 0.1078 | +0.0164 |
| date | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| date | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| date | hit@5 | 0.1250 | 0.1250 | +0.0000 |
| date | recall@5 | 0.1250 | 0.1250 | +0.0000 |
| date | recall@10 | 0.2500 | 0.2750 | +0.0250 |
| date | recall@25 | 0.8250 | 1.0000 | +0.1750 |
| exact_code | n | 40.0000 | 40.0000 | +0.0000 |
| exact_code | MRR | 0.2614 | 0.4125 | +0.1511 |
| exact_code | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| exact_code | hit@3 | 0.4000 | 1.0000 | +0.6000 |
| exact_code | hit@5 | 0.4500 | 1.0000 | +0.5500 |
| exact_code | recall@5 | 0.4500 | 1.0000 | +0.5500 |
| exact_code | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| fallback | n | 40.0000 | 40.0000 | +0.0000 |
| fallback | MRR | 0.0284 | 0.0284 | +0.0000 |
| fallback | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@25 | 0.0000 | 0.0000 | +0.0000 |
| old_vs_recent | n | 40.0000 | 40.0000 | +0.0000 |
| old_vs_recent | MRR | 0.8387 | 0.8387 | +0.0000 |
| old_vs_recent | hit@1 | 0.8000 | 0.8000 | +0.0000 |
| old_vs_recent | hit@3 | 0.8000 | 0.8000 | +0.0000 |
| old_vs_recent | hit@5 | 0.9250 | 0.9250 | +0.0000 |
| old_vs_recent | recall@5 | 0.9250 | 0.9250 | +0.0000 |
| old_vs_recent | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| overall | n | 240.0000 | 240.0000 | +0.0000 |
| overall | MRR | 0.2886 | 0.3166 | +0.0279 |
| overall | hit@1 | 0.1333 | 0.1333 | +0.0000 |
| overall | hit@3 | 0.3667 | 0.4667 | +0.1000 |
| overall | hit@5 | 0.4167 | 0.5083 | +0.0917 |
| overall | recall@5 | 0.4167 | 0.5083 | +0.0917 |
| overall | recall@10 | 0.5417 | 0.5458 | +0.0042 |
| overall | recall@25 | 0.8042 | 0.8333 | +0.0292 |
| paraphrase | n | 40.0000 | 40.0000 | +0.0000 |
| paraphrase | MRR | 0.0453 | 0.0453 | +0.0000 |
| paraphrase | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| plain | n | 40.0000 | 40.0000 | +0.0000 |
| plain | MRR | 0.4667 | 0.4667 | +0.0000 |
| plain | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| plain | hit@3 | 1.0000 | 1.0000 | +0.0000 |
| plain | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@25 | 1.0000 | 1.0000 | +0.0000 |

| kind | pairs | better | worse | same | entered 5 | left 5 | entered 10 | left 10 | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 240 | 53 | 0 | 187 | 22 | 0 | 1 | 0 | 2.22045e-16 |
| date | 40 | 29 | 0 | 11 | 0 | 0 | 1 | 0 | 3.72529e-09 |
| exact_code | 40 | 24 | 0 | 16 | 22 | 0 | 0 | 0 | 1.19209e-07 |
| fallback | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| old_vs_recent | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| paraphrase | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| plain | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |

### 30 largest regressions (reciprocal-rank change)


### 30 largest improvements (reciprocal-rank change)

- exact_code-01 (exact_code): What is the production IP address for Birch 1? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"6": {"raw_vector_score": 0.7618228261128426, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"6": {"raw_vector_score": 0.7618228261128426, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-06 (exact_code): What is the production IP address for Birch 2? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"31": {"raw_vector_score": 0.7735182322664895, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"31": {"raw_vector_score": 0.7735182322664895, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-07 (exact_code): What is the production version for Maple 2? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"36": {"raw_vector_score": 0.6763831960633923, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"36": {"raw_vector_score": 0.6763831960633923, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-11 (exact_code): What is the production IP address for Birch 3? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"56": {"raw_vector_score": 0.7775198389993706, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"56": {"raw_vector_score": 0.7775198389993706, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-22 (exact_code): What is the production version for Maple 5? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"111": {"raw_vector_score": 0.6570919992070177, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"111": {"raw_vector_score": 0.6570919992070177, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-13 (exact_code): What is the production ticket id for Willow 3? Rank 6 → 2; ΔRR +0.3333. Gold flags: `{"66": {"raw_vector_score": 0.7854917273962833, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"66": {"raw_vector_score": 0.7854917273962833, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-17 (exact_code): What is the production version for Maple 4? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"86": {"raw_vector_score": 0.6730195449552806, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"86": {"raw_vector_score": 0.6730195449552806, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-02 (exact_code): What is the production version for Maple 1? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"11": {"raw_vector_score": 0.6569053656070323, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"11": {"raw_vector_score": 0.6569053656070323, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-12 (exact_code): What is the production version for Maple 3? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"61": {"raw_vector_score": 0.6570099585936364, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"61": {"raw_vector_score": 0.6570099585936364, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-16 (exact_code): What is the production IP address for Birch 4? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"81": {"raw_vector_score": 0.7342829047390795, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"81": {"raw_vector_score": 0.7342829047390795, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-21 (exact_code): What is the production IP address for Birch 5? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"106": {"raw_vector_score": 0.7456620474518352, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"106": {"raw_vector_score": 0.7456620474518352, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-26 (exact_code): What is the production IP address for Birch 6? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"131": {"raw_vector_score": 0.7577306072026975, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"131": {"raw_vector_score": 0.7577306072026975, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-27 (exact_code): What is the production version for Maple 6? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"136": {"raw_vector_score": 0.6717250290726615, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"136": {"raw_vector_score": 0.6717250290726615, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-32 (exact_code): What is the production version for Maple 7? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"161": {"raw_vector_score": 0.6727222108103602, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"161": {"raw_vector_score": 0.6727222108103602, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-36 (exact_code): What is the production IP address for Birch 8? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"181": {"raw_vector_score": 0.7456679153222714, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"181": {"raw_vector_score": 0.7456679153222714, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-37 (exact_code): What is the production version for Maple 8? Rank 10 → 3; ΔRR +0.2333. Gold flags: `{"186": {"raw_vector_score": 0.6729806631587606, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"186": {"raw_vector_score": 0.6729806631587606, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-03 (exact_code): What is the production ticket id for Willow 1? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"16": {"raw_vector_score": 0.7520449747471538, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"16": {"raw_vector_score": 0.7520449747471538, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-08 (exact_code): What is the production ticket id for Willow 2? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"41": {"raw_vector_score": 0.771409500391902, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"41": {"raw_vector_score": 0.771409500391902, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-18 (exact_code): What is the production ticket id for Willow 4? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"91": {"raw_vector_score": 0.7647099558696637, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"91": {"raw_vector_score": 0.7647099558696637, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-23 (exact_code): What is the production ticket id for Willow 5? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"116": {"raw_vector_score": 0.7741821168454741, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"116": {"raw_vector_score": 0.7741821168454741, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-28 (exact_code): What is the production ticket id for Willow 6? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"141": {"raw_vector_score": 0.7815221642607467, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"141": {"raw_vector_score": 0.7815221642607467, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-33 (exact_code): What is the production ticket id for Willow 7? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"166": {"raw_vector_score": 0.7755828725672221, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"166": {"raw_vector_score": 0.7755828725672221, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-38 (exact_code): What is the production ticket id for Willow 8? Rank 7 → 3; ΔRR +0.1905. Gold flags: `{"191": {"raw_vector_score": 0.7727208874432718, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"191": {"raw_vector_score": 0.7727208874432718, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-31 (exact_code): What is the production IP address for Birch 7? Rank 4 → 3; ΔRR +0.0833. Gold flags: `{"156": {"raw_vector_score": 0.7601758302010728, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}` → `{"156": {"raw_vector_score": 0.7601758302010728, "flags": ["exact_rescue"], "access_boost": 1.0, "freshness": 0.8}}`
- date-01 (date): What did the Birch 1 committee decide in April 30, 2025? Rank 26 → 11; ΔRR +0.0524. Gold flags: `{"204": {"raw_vector_score": 0.6992148427529135, "access_boost": 1.0, "freshness": 0.0}}` → `{"204": {"raw_vector_score": 0.6992148427529135, "access_boost": 1.0, "freshness": 0.0}}`
- date-09 (date): What did the Elm 2 committee decide in April 30, 2025? Rank 26 → 11; ΔRR +0.0524. Gold flags: `{"228": {"raw_vector_score": 0.700921420687245, "access_boost": 1.0, "freshness": 0.0}}` → `{"228": {"raw_vector_score": 0.700921420687245, "access_boost": 1.0, "freshness": 0.0}}`
- date-33 (date): What did the Willow 7 committee decide in April 30, 2025? Rank 26 → 11; ΔRR +0.0524. Gold flags: `{"300": {"raw_vector_score": 0.7212586864021862, "access_boost": 1.0, "freshness": 0.0}}` → `{"300": {"raw_vector_score": 0.7212586864021862, "access_boost": 1.0, "freshness": 0.0}}`
- date-05 (date): What did the Cedar 2 committee decide in April 30, 2025? Rank 25 → 11; ΔRR +0.0509. Gold flags: `{"216": {"raw_vector_score": 0.699927129465641, "access_boost": 1.0, "freshness": 0.0}}` → `{"216": {"raw_vector_score": 0.699927129465641, "access_boost": 1.0, "freshness": 0.0}}`
- date-25 (date): What did the Cedar 6 committee decide in April 30, 2025? Rank 25 → 11; ΔRR +0.0509. Gold flags: `{"276": {"raw_vector_score": 0.6932717546238139, "access_boost": 1.0, "freshness": 0.0}}` → `{"276": {"raw_vector_score": 0.6932717546238139, "access_boost": 1.0, "freshness": 0.0}}`
- date-13 (date): What did the Willow 3 committee decide in April 30, 2025? Rank 27 → 12; ΔRR +0.0463. Gold flags: `{"240": {"raw_vector_score": 0.7191375987017012, "access_boost": 1.0, "freshness": 0.0}}` → `{"240": {"raw_vector_score": 0.7191375987017012, "access_boost": 1.0, "freshness": 0.0}}`

Unpaired/error queries: 0. Recorded skips: first 0, second 0.

## full_fresh_state / full_fresh_state

| kind | metric | first | second | difference |
|---|---|---:|---:|---:|
| date | n | 40.0000 | 40.0000 | +0.0000 |
| date | MRR | 0.0914 | 0.1078 | +0.0164 |
| date | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| date | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| date | hit@5 | 0.1250 | 0.1250 | +0.0000 |
| date | recall@5 | 0.1250 | 0.1250 | +0.0000 |
| date | recall@10 | 0.2500 | 0.2750 | +0.0250 |
| date | recall@25 | 0.8250 | 1.0000 | +0.1750 |
| exact_code | n | 40.0000 | 40.0000 | +0.0000 |
| exact_code | MRR | 0.4654 | 0.6625 | +0.1971 |
| exact_code | hit@1 | 0.2000 | 0.3250 | +0.1250 |
| exact_code | hit@3 | 0.6500 | 1.0000 | +0.3500 |
| exact_code | hit@5 | 0.8750 | 1.0000 | +0.1250 |
| exact_code | recall@5 | 0.8750 | 1.0000 | +0.1250 |
| exact_code | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| fallback | n | 40.0000 | 40.0000 | +0.0000 |
| fallback | MRR | 0.0284 | 0.0284 | +0.0000 |
| fallback | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@25 | 0.0000 | 0.0000 | +0.0000 |
| old_vs_recent | n | 40.0000 | 40.0000 | +0.0000 |
| old_vs_recent | MRR | 0.8387 | 0.8387 | +0.0000 |
| old_vs_recent | hit@1 | 0.8000 | 0.8000 | +0.0000 |
| old_vs_recent | hit@3 | 0.8000 | 0.8000 | +0.0000 |
| old_vs_recent | hit@5 | 0.9250 | 0.9250 | +0.0000 |
| old_vs_recent | recall@5 | 0.9250 | 0.9250 | +0.0000 |
| old_vs_recent | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| overall | n | 240.0000 | 240.0000 | +0.0000 |
| overall | MRR | 0.3226 | 0.3582 | +0.0356 |
| overall | hit@1 | 0.1667 | 0.1875 | +0.0208 |
| overall | hit@3 | 0.4083 | 0.4667 | +0.0583 |
| overall | hit@5 | 0.4875 | 0.5083 | +0.0208 |
| overall | recall@5 | 0.4875 | 0.5083 | +0.0208 |
| overall | recall@10 | 0.5417 | 0.5458 | +0.0042 |
| overall | recall@25 | 0.8042 | 0.8333 | +0.0292 |
| paraphrase | n | 40.0000 | 40.0000 | +0.0000 |
| paraphrase | MRR | 0.0453 | 0.0453 | +0.0000 |
| paraphrase | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| plain | n | 40.0000 | 40.0000 | +0.0000 |
| plain | MRR | 0.4667 | 0.4667 | +0.0000 |
| plain | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| plain | hit@3 | 1.0000 | 1.0000 | +0.0000 |
| plain | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@25 | 1.0000 | 1.0000 | +0.0000 |

| kind | pairs | better | worse | same | entered 5 | left 5 | entered 10 | left 10 | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 240 | 54 | 0 | 186 | 5 | 0 | 1 | 0 | 1.11022e-16 |
| date | 40 | 29 | 0 | 11 | 0 | 0 | 1 | 0 | 3.72529e-09 |
| exact_code | 40 | 25 | 0 | 15 | 5 | 0 | 0 | 0 | 5.96046e-08 |
| fallback | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| old_vs_recent | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| paraphrase | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| plain | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |

### 30 largest regressions (reciprocal-rank change)


### 30 largest improvements (reciprocal-rank change)

- exact_code-07 (exact_code): What is the production version for Maple 2? Rank 3 → 1; ΔRR +0.6667. Gold flags: `{"36": {"raw_vector_score": 0.6763831960633923, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"36": {"raw_vector_score": 0.6763831960633923, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-17 (exact_code): What is the production version for Maple 4? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"86": {"raw_vector_score": 0.6730195449552806, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"86": {"raw_vector_score": 0.6730195449552806, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-22 (exact_code): What is the production version for Maple 5? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"111": {"raw_vector_score": 0.6570919992070177, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"111": {"raw_vector_score": 0.6570919992070177, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-25 (exact_code): Какой port у сервиса Cedar 6? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"126": {"raw_vector_score": 0.7245929969202753, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"126": {"raw_vector_score": 0.7245929969202753, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-30 (exact_code): Какой port у сервиса Cedar 7? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"151": {"raw_vector_score": 0.732355313509159, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"151": {"raw_vector_score": 0.732355313509159, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-02 (exact_code): What is the production version for Maple 1? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"11": {"raw_vector_score": 0.6569053656070323, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"11": {"raw_vector_score": 0.6569053656070323, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-32 (exact_code): What is the production version for Maple 7? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"161": {"raw_vector_score": 0.6727222108103602, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"161": {"raw_vector_score": 0.6727222108103602, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-03 (exact_code): What is the production ticket id for Willow 1? Rank 8 → 2; ΔRR +0.3750. Gold flags: `{"16": {"raw_vector_score": 0.7520449747471538, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"16": {"raw_vector_score": 0.7520449747471538, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-01 (exact_code): What is the production IP address for Birch 1? Rank 7 → 2; ΔRR +0.3571. Gold flags: `{"6": {"raw_vector_score": 0.7618228261128426, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"6": {"raw_vector_score": 0.7618228261128426, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-33 (exact_code): What is the production ticket id for Willow 7? Rank 7 → 2; ΔRR +0.3571. Gold flags: `{"166": {"raw_vector_score": 0.7755828725672221, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"166": {"raw_vector_score": 0.7755828725672221, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-18 (exact_code): What is the production ticket id for Willow 4? Rank 5 → 2; ΔRR +0.3000. Gold flags: `{"91": {"raw_vector_score": 0.7647099558696637, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"91": {"raw_vector_score": 0.7647099558696637, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-27 (exact_code): What is the production version for Maple 6? Rank 5 → 2; ΔRR +0.3000. Gold flags: `{"136": {"raw_vector_score": 0.6717250290726615, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"136": {"raw_vector_score": 0.6717250290726615, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-06 (exact_code): What is the production IP address for Birch 2? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"31": {"raw_vector_score": 0.7735182322664895, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"31": {"raw_vector_score": 0.7735182322664895, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-16 (exact_code): What is the production IP address for Birch 4? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"81": {"raw_vector_score": 0.7342829047390795, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"81": {"raw_vector_score": 0.7342829047390795, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-21 (exact_code): What is the production IP address for Birch 5? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"106": {"raw_vector_score": 0.7456620474518352, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"106": {"raw_vector_score": 0.7456620474518352, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-26 (exact_code): What is the production IP address for Birch 6? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"131": {"raw_vector_score": 0.7577306072026975, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"131": {"raw_vector_score": 0.7577306072026975, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-28 (exact_code): What is the production ticket id for Willow 6? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"141": {"raw_vector_score": 0.7815221642607467, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"141": {"raw_vector_score": 0.7815221642607467, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-36 (exact_code): What is the production IP address for Birch 8? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"181": {"raw_vector_score": 0.7456679153222714, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"181": {"raw_vector_score": 0.7456679153222714, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-38 (exact_code): What is the production ticket id for Willow 8? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"191": {"raw_vector_score": 0.7727208874432718, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"191": {"raw_vector_score": 0.7727208874432718, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-08 (exact_code): What is the production ticket id for Willow 2? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"41": {"raw_vector_score": 0.771409500391902, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"41": {"raw_vector_score": 0.771409500391902, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-11 (exact_code): What is the production IP address for Birch 3? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"56": {"raw_vector_score": 0.7775198389993706, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"56": {"raw_vector_score": 0.7775198389993706, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-12 (exact_code): What is the production version for Maple 3? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"61": {"raw_vector_score": 0.6570099585936364, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"61": {"raw_vector_score": 0.6570099585936364, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-13 (exact_code): What is the production ticket id for Willow 3? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"66": {"raw_vector_score": 0.7854917273962833, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"66": {"raw_vector_score": 0.7854917273962833, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-23 (exact_code): What is the production ticket id for Willow 5? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"116": {"raw_vector_score": 0.7741821168454741, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"116": {"raw_vector_score": 0.7741821168454741, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-31 (exact_code): What is the production IP address for Birch 7? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"156": {"raw_vector_score": 0.7601758302010728, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"156": {"raw_vector_score": 0.7601758302010728, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- date-01 (date): What did the Birch 1 committee decide in April 30, 2025? Rank 26 → 11; ΔRR +0.0524. Gold flags: `{"204": {"raw_vector_score": 0.6992148427529135, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}` → `{"204": {"raw_vector_score": 0.6992148427529135, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-09 (date): What did the Elm 2 committee decide in April 30, 2025? Rank 26 → 11; ΔRR +0.0524. Gold flags: `{"228": {"raw_vector_score": 0.700921420687245, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}` → `{"228": {"raw_vector_score": 0.700921420687245, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-33 (date): What did the Willow 7 committee decide in April 30, 2025? Rank 26 → 11; ΔRR +0.0524. Gold flags: `{"300": {"raw_vector_score": 0.7212586864021862, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}` → `{"300": {"raw_vector_score": 0.7212586864021862, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-05 (date): What did the Cedar 2 committee decide in April 30, 2025? Rank 25 → 11; ΔRR +0.0509. Gold flags: `{"216": {"raw_vector_score": 0.699927129465641, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}` → `{"216": {"raw_vector_score": 0.699927129465641, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-25 (date): What did the Cedar 6 committee decide in April 30, 2025? Rank 25 → 11; ΔRR +0.0509. Gold flags: `{"276": {"raw_vector_score": 0.6932717546238139, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}` → `{"276": {"raw_vector_score": 0.6932717546238139, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`

Unpaired/error queries: 0. Recorded skips: first 0, second 0.

## recall_flow / recall_flow

| kind | metric | first | second | difference |
|---|---|---:|---:|---:|
| date | n | 40.0000 | 40.0000 | +0.0000 |
| date | MRR | 0.0848 | 0.1078 | +0.0230 |
| date | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| date | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| date | hit@5 | 0.1250 | 0.1250 | +0.0000 |
| date | recall@5 | 0.1250 | 0.1250 | +0.0000 |
| date | recall@10 | 0.2500 | 0.2750 | +0.0250 |
| date | recall@25 | 0.8250 | 1.0000 | +0.1750 |
| exact_code | n | 40.0000 | 40.0000 | +0.0000 |
| exact_code | MRR | 0.4654 | 0.6625 | +0.1971 |
| exact_code | hit@1 | 0.2000 | 0.3250 | +0.1250 |
| exact_code | hit@3 | 0.6500 | 1.0000 | +0.3500 |
| exact_code | hit@5 | 0.8750 | 1.0000 | +0.1250 |
| exact_code | recall@5 | 0.8750 | 1.0000 | +0.1250 |
| exact_code | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| exact_code | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| fallback | n | 40.0000 | 40.0000 | +0.0000 |
| fallback | MRR | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| fallback | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| fallback | recall@25 | 0.0000 | 0.0000 | +0.0000 |
| old_vs_recent | n | 40.0000 | 40.0000 | +0.0000 |
| old_vs_recent | MRR | 0.8387 | 0.8387 | +0.0000 |
| old_vs_recent | hit@1 | 0.8000 | 0.8000 | +0.0000 |
| old_vs_recent | hit@3 | 0.8000 | 0.8000 | +0.0000 |
| old_vs_recent | hit@5 | 0.9250 | 0.9250 | +0.0000 |
| old_vs_recent | recall@5 | 0.9250 | 0.9250 | +0.0000 |
| old_vs_recent | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| old_vs_recent | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| overall | n | 240.0000 | 240.0000 | +0.0000 |
| overall | MRR | 0.3168 | 0.3535 | +0.0367 |
| overall | hit@1 | 0.1667 | 0.1875 | +0.0208 |
| overall | hit@3 | 0.4083 | 0.4667 | +0.0583 |
| overall | hit@5 | 0.4875 | 0.5083 | +0.0208 |
| overall | recall@5 | 0.4875 | 0.5083 | +0.0208 |
| overall | recall@10 | 0.5417 | 0.5458 | +0.0042 |
| overall | recall@25 | 0.8042 | 0.8333 | +0.0292 |
| paraphrase | n | 40.0000 | 40.0000 | +0.0000 |
| paraphrase | MRR | 0.0453 | 0.0453 | +0.0000 |
| paraphrase | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@3 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | hit@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@5 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@10 | 0.0000 | 0.0000 | +0.0000 |
| paraphrase | recall@25 | 1.0000 | 1.0000 | +0.0000 |
| plain | n | 40.0000 | 40.0000 | +0.0000 |
| plain | MRR | 0.4667 | 0.4667 | +0.0000 |
| plain | hit@1 | 0.0000 | 0.0000 | +0.0000 |
| plain | hit@3 | 1.0000 | 1.0000 | +0.0000 |
| plain | hit@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| plain | recall@25 | 1.0000 | 1.0000 | +0.0000 |

| kind | pairs | better | worse | same | entered 5 | left 5 | entered 10 | left 10 | sign p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 240 | 54 | 0 | 186 | 5 | 0 | 1 | 0 | 1.11022e-16 |
| date | 40 | 29 | 0 | 11 | 0 | 0 | 1 | 0 | 3.72529e-09 |
| exact_code | 40 | 25 | 0 | 15 | 5 | 0 | 0 | 0 | 5.96046e-08 |
| fallback | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| old_vs_recent | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| paraphrase | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |
| plain | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0 | 1 |

### 30 largest regressions (reciprocal-rank change)


### 30 largest improvements (reciprocal-rank change)

- exact_code-07 (exact_code): What is the production version for Maple 2? Rank 3 → 1; ΔRR +0.6667. Gold flags: `{"36": {"raw_vector_score": 0.6763831960633923, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"36": {"raw_vector_score": 0.6763831960633923, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-17 (exact_code): What is the production version for Maple 4? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"86": {"raw_vector_score": 0.6730195449552806, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"86": {"raw_vector_score": 0.6730195449552806, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-22 (exact_code): What is the production version for Maple 5? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"111": {"raw_vector_score": 0.6570919992070177, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"111": {"raw_vector_score": 0.6570919992070177, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-25 (exact_code): Какой port у сервиса Cedar 6? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"126": {"raw_vector_score": 0.7245929969202753, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"126": {"raw_vector_score": 0.7245929969202753, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-30 (exact_code): Какой port у сервиса Cedar 7? Rank 2 → 1; ΔRR +0.5000. Gold flags: `{"151": {"raw_vector_score": 0.732355313509159, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"151": {"raw_vector_score": 0.732355313509159, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-02 (exact_code): What is the production version for Maple 1? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"11": {"raw_vector_score": 0.6569053656070323, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"11": {"raw_vector_score": 0.6569053656070323, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-32 (exact_code): What is the production version for Maple 7? Rank 9 → 2; ΔRR +0.3889. Gold flags: `{"161": {"raw_vector_score": 0.6727222108103602, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"161": {"raw_vector_score": 0.6727222108103602, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-03 (exact_code): What is the production ticket id for Willow 1? Rank 8 → 2; ΔRR +0.3750. Gold flags: `{"16": {"raw_vector_score": 0.7520449747471538, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"16": {"raw_vector_score": 0.7520449747471538, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-01 (exact_code): What is the production IP address for Birch 1? Rank 7 → 2; ΔRR +0.3571. Gold flags: `{"6": {"raw_vector_score": 0.7618228261128426, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"6": {"raw_vector_score": 0.7618228261128426, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-33 (exact_code): What is the production ticket id for Willow 7? Rank 7 → 2; ΔRR +0.3571. Gold flags: `{"166": {"raw_vector_score": 0.7755828725672221, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"166": {"raw_vector_score": 0.7755828725672221, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-18 (exact_code): What is the production ticket id for Willow 4? Rank 5 → 2; ΔRR +0.3000. Gold flags: `{"91": {"raw_vector_score": 0.7647099558696637, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"91": {"raw_vector_score": 0.7647099558696637, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-27 (exact_code): What is the production version for Maple 6? Rank 5 → 2; ΔRR +0.3000. Gold flags: `{"136": {"raw_vector_score": 0.6717250290726615, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"136": {"raw_vector_score": 0.6717250290726615, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-06 (exact_code): What is the production IP address for Birch 2? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"31": {"raw_vector_score": 0.7735182322664895, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"31": {"raw_vector_score": 0.7735182322664895, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-16 (exact_code): What is the production IP address for Birch 4? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"81": {"raw_vector_score": 0.7342829047390795, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"81": {"raw_vector_score": 0.7342829047390795, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-21 (exact_code): What is the production IP address for Birch 5? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"106": {"raw_vector_score": 0.7456620474518352, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"106": {"raw_vector_score": 0.7456620474518352, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-26 (exact_code): What is the production IP address for Birch 6? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"131": {"raw_vector_score": 0.7577306072026975, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"131": {"raw_vector_score": 0.7577306072026975, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-28 (exact_code): What is the production ticket id for Willow 6? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"141": {"raw_vector_score": 0.7815221642607467, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"141": {"raw_vector_score": 0.7815221642607467, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-36 (exact_code): What is the production IP address for Birch 8? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"181": {"raw_vector_score": 0.7456679153222714, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"181": {"raw_vector_score": 0.7456679153222714, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-38 (exact_code): What is the production ticket id for Willow 8? Rank 4 → 2; ΔRR +0.2500. Gold flags: `{"191": {"raw_vector_score": 0.7727208874432718, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"191": {"raw_vector_score": 0.7727208874432718, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-08 (exact_code): What is the production ticket id for Willow 2? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"41": {"raw_vector_score": 0.771409500391902, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"41": {"raw_vector_score": 0.771409500391902, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-11 (exact_code): What is the production IP address for Birch 3? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"56": {"raw_vector_score": 0.7775198389993706, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"56": {"raw_vector_score": 0.7775198389993706, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-12 (exact_code): What is the production version for Maple 3? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"61": {"raw_vector_score": 0.6570099585936364, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"61": {"raw_vector_score": 0.6570099585936364, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-13 (exact_code): What is the production ticket id for Willow 3? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"66": {"raw_vector_score": 0.7854917273962833, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"66": {"raw_vector_score": 0.7854917273962833, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-23 (exact_code): What is the production ticket id for Willow 5? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"116": {"raw_vector_score": 0.7741821168454741, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"116": {"raw_vector_score": 0.7741821168454741, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- exact_code-31 (exact_code): What is the production IP address for Birch 7? Rank 3 → 2; ΔRR +0.1667. Gold flags: `{"156": {"raw_vector_score": 0.7601758302010728, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}` → `{"156": {"raw_vector_score": 0.7601758302010728, "flags": ["exact_rescue"], "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.8}}`
- date-01 (date): What did the Birch 1 committee decide in April 30, 2025? Rank None → 11; ΔRR +0.0909. Gold flags: `{}` → `{"204": {"raw_vector_score": 0.6992148427529135, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-09 (date): What did the Elm 2 committee decide in April 30, 2025? Rank None → 11; ΔRR +0.0909. Gold flags: `{}` → `{"228": {"raw_vector_score": 0.700921420687245, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-33 (date): What did the Willow 7 committee decide in April 30, 2025? Rank None → 11; ΔRR +0.0909. Gold flags: `{}` → `{"300": {"raw_vector_score": 0.7212586864021862, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-13 (date): What did the Willow 3 committee decide in April 30, 2025? Rank None → 12; ΔRR +0.0833. Gold flags: `{}` → `{"240": {"raw_vector_score": 0.7191375987017012, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`
- date-21 (date): What did the Birch 5 committee decide in April 30, 2025? Rank None → 12; ΔRR +0.0833. Gold flags: `{}` → `{"264": {"raw_vector_score": 0.72112398127049, "q_value": 0.6, "curiosity_boosted": true, "access_boost": 1.0, "freshness": 0.0}}`

Unpaired/error queries: 0. Recorded skips: first 0, second 0.

