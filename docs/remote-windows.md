# Window chunks for memories written through the API

`Ingestor(window_size=N)` and `mnemostack index --window-size N` store one
extra chunk for every N neighbouring items of a source. The extra chunk carries
the context of adjacent turns, which helps recall on chat transcripts.

Memories written through `POST /memories` and the MCP tool
`mnemostack_remember` arrive one at a time, so the service builds the same
chunks from the memories it already holds. This is off by default.

## Turning it on

```yaml
ingest:
  remote_window_size: 3   # 1 = off (default)
```

or `MNEMOSTACK_REMOTE_WINDOW_SIZE=3`. The setting is separate from
`vector.window_size`, which only affects file indexing. Sizes 2 and 3 are the
tested ones.

Turning it on changes later writes only. For memories that are already stored,
run `mnemostack rebuild-windows` (below).

## What the caller must send

- `source` names the conversation. A window never joins memories of two
  sources or of two tenants. A memory with an empty `source` gets no window.
- `offset` gives the order inside the source. Send a distinct, increasing
  `offset` for every message. A caller that leaves every `offset` at the
  default `0` gets no windows.
- `chunk: true` documents get windows over their chunks by the same rule.

For eligible memories (see Rules) in the same order, the stored window chunks
have the content of the ones `Ingestor(window_size=N)` stores: text, member
order, timestamps, tags and metadata. Two differences. The point id is
another one: API-built windows have their own id space, so a window and a
memory can never share an id. And with a custom `recall.text_key` the window
chunk carries its own text under that key. A window chunk carries the metadata, tags and timestamp of its middle
member, plus `window_start_ts` / `window_end_ts`.

## Rules

- Order comes from `offset`, never from arrival time. Messages written out of
  order end in the same windows as messages written in order: a window that
  joins two memories that are no longer neighbours is deleted.
- Two live memories of one source with the same `offset` have no defined
  order. No window includes that offset, and no window bridges it. The write
  response counts such offsets in `windows.ambiguous_offsets`. Writing new text
  at an existing offset therefore removes the windows that held the old text;
  retract or delete the old memory to make the offset usable again.
- A retraction or deletion only removes windows. The windows across the gap
  it leaves are built by a later write of a neighbour, or by
  `mnemostack rebuild-windows`.
- With `ingest.secret_policy: block` the joined text of a window is checked
  too: a window that trips the policy is not built, and the memories stay.
  The write response counts it in `windows.blocked_by_secret_policy`.
  `rebuild-windows` also removes stored windows that trip the current policy,
  and its report counts them. Changing the policy alone removes nothing; the
  cleanup happens on a rebuild.
- A memory with `valid_from` / `valid_until`, or one that belongs to an
  `index_root`, stays out of windows and is not bridged either. Its text must
  not stay readable through a chunk with another validity period.
- Retracting (`POST /invalidate`, `mnemostack_invalidate`, by id) or deleting
  (`DELETE /memories`, by id) a memory deletes every window that holds it, in
  the same call, also after the setting was turned off again. The CLI
  `mnemostack invalidate` does the same. A source-scoped call does it for
  the memories it retracts or deletes, batch by batch. A source-scoped
  `POST /invalidate` counts memories only in `requested` and `invalidated`,
  and a batch that held only windows answers `requested: 0` with
  `complete: false`. A source-scoped `DELETE /memories` counts the window
  chunks it deletes too.
- Window chunks are derived data. They are deleted, not retracted, so recall
  with `include_invalidated` or `as_of` returns a retracted memory without its
  windows. Writing the retracted memory again builds them again, unless the
  retraction set a `valid_until`: that bound stays, and it keeps the memory
  out of windows.
- A source that holds window chunks written by `mnemostack index` or
  `Ingestor` is left to that writer: the service builds no windows there,
  removes the ones it built there before, and
  the write response counts it in `windows.skipped_sources`. Such a source is
  recognised by a window chunk without the member list that every API-built
  window carries. The service writes a window chunk and its member list in
  one operation, so a failed window step cannot make a source look like one
  of the other writer's.
- After a change of the size, later writes build windows of the new size.
  Windows of the old size stay until `mnemostack rebuild-windows`.
- `GET /memories?source=…` lists the window chunks of the source together with
  the memories.
- `mnemostack resolve` cannot verify a window chunk (its text is a
  concatenation); resolve the member memories instead.

## The write response

`POST /memories` and `mnemostack_remember` report the window step in `windows`
(`null` / absent while the setting is off):

```json
{"created": 1, "removed": 0, "ambiguous_offsets": 0, "skipped_sources": 0,
 "blocked_by_secret_policy": 0, "complete": true, "reason": null}
```

The memories of a request are stored first, under the existing rules. The
window step never fails them. When a window could not be written, `complete` is
`false` and `reason` says why:

| `reason` | Meaning |
| --- | --- |
| `quota` | The windows would pass the tenant's `max_points`. Window chunks count against the quota. |
| `embedding` | The embedding provider failed for a window text. |
| `error` | The store failed during the window step. |
| `unsupported` | A custom store without `scroll`. |

Sending the same request again repairs the missing windows (the memories come
back as `duplicate` and cost no embedding). `mnemostack rebuild-windows` does
the same for a whole source or tenant.

## Costs

- Storage: about two times the points for any size (one window per memory,
  less N-1 per source). The stored text is about three times as large with
  size 2 and four times with size 3. Both the vector store and the lexical
  index grow.
- Embedding: an append to a source that already holds N-1 memories embeds
  one extra text of N joined messages. A memory written between stored ones
  can belong to up to N new windows. A rebuild embeds every missing window
  of the source.
- Reads: every write, and every retraction or deletion, reads the ids and
  offsets of all points of the source, inside the tenant's write lock. The
  time of a write therefore grows with the length of the conversation, and
  other writes of the tenant wait for it. One measurement on a Qdrant 1.20
  server over a local network, with a trivial embedder and size 3: a write
  took about 90 ms into a source of 100 memories, 150 ms at 1,000 and 800 ms
  at 10,000, against about 15 ms without windows. Keep a source bounded, for
  example one per conversation or per day. With the setting off, only a
  retraction of a memory that is a window member reads the source.
- New collections have a keyword index on `source`. A collection created
  before that, or by another tool, is scanned whole for each read; create
  the index with `mnemostack payload-index source --schema keyword`.
- Ranking: window chunks compete with single memories for the result slots. On
  a development subset of LongMemEval, windows of 3 lowered the number of
  knowledge-update questions with the newer session ranked first from 21 to 17
  of 24. This effect is not measured for size 2.

## Rebuild

```bash
mnemostack rebuild-windows --all-sources --dry-run
mnemostack rebuild-windows --tenant acme --source chat/2026-08-19
```

The scope is explicit: `--source` (repeatable) or `--all-sources`. A stored
memory does not record which surface wrote it. The command, like the write
path, treats every memory that has a `source` and no `index_root` as a
neighbour, also one that an SDK `Ingestor` wrote without windows. Check the
counts of a dry run before `--all-sources` on a collection that such a writer
shares.

For every selected source the command deletes the API-built windows that are
no longer valid or have another size, and writes the missing ones. It is
idempotent, and it reads only live memories, so it never brings back retracted
or deleted text. `--window-size 1` removes the API-built windows. Window chunks
written by `mnemostack index` or `Ingestor` are not touched.

The report counts what was skipped: ambiguous offsets, memories with a validity
bound or an `index_root`, memories without a `source`, and sources with window
chunks of `index` or `Ingestor`. It also counts the windows that the secret
policy blocks, without their text. A dry run applies the policy too: it reads
the member texts and writes nothing.

## Concurrency

Writes and retractions of one tenant are serialized inside one server process,
except a source-scoped call of an unscoped caller that touches a tenant's
memories. The lock is a lock of the process: several server processes or
replicas, an MCP server next to the HTTP server, and a running
`rebuild-windows` do not share it. Each write checks the source again after
it stored its windows, and each retraction checks again after it took effect,
so two interleaved calls do not leave a window over retracted text or over
memories that are no longer neighbours. Such a race can leave a window missing;
`rebuild-windows` restores it. This rests on the order of the calls and on
tests that interleave a second writer inside one process; it is not tested
with separate server processes under load.
