import test from "node:test";
import assert from "node:assert/strict";
import { createBeforePromptBuildHandler, createRuntime, mapJsonToResult } from "../index.js";

for (const routineTags of [[], ["routine"]]) test(`default hook injects a healthy stock answer (${routineTags.length} routine notes)`, async () => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url, body: JSON.parse(init.body) });
    return new Response(JSON.stringify({ answer: "We selected the blue connector.", confidence: 0.9, degraded: routineTags, notes: routineTags, sources: [] }), { status: 200 });
  };
  const runtime = createRuntime({}, { info() {}, warn() {} });
  try {
    assert.equal(runtime.cfg.allowDegraded, false);
    const result = await createBeforePromptBuildHandler(runtime)(
      { prompt: "what did we decide?" }, { chatType: "direct", sessionKey: "synthetic" }
    );
    assert.match(result?.prependContext || "", /blue connector/);
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, runtime.cfg.backends[0].url);
    assert.equal(requests[0].body.query, "what did we decide?");
  } finally {
    await runtime.dispose();
    globalThis.fetch = originalFetch;
  }
});

for (const [degraded, status] of [[[], "ok"], [["vector"], "degraded"], [false, "ok"], [true, "degraded"], [undefined, "ok"]]) {
  test(`degraded mapping preserves ${JSON.stringify(degraded)}`, () => {
    const result = mapJsonToResult({ answer: "fact", confidence: 0.9, degraded });
    assert.equal(result.status, status);
    assert.equal(result.answer, "fact");
  });
}

test("explicit status and custom nested degraded mapping remain supported", () => {
  assert.equal(mapJsonToResult({ answer: "fact", status: "ok", degraded: ["vector"] }).status, "ok");
  assert.equal(mapJsonToResult({ answer: "fact", health: { failed: [] } }, { degradedPath: "health.failed" }).status, "ok");
  assert.equal(mapJsonToResult({ degraded: [] }).status, "not_found");
});

test("routine notes are not component failures in the stock answer contract", () => {
  const routine = { answer: "fact", confidence: 0.9, degraded: ["routine"], notes: ["routine"] };
  assert.equal(mapJsonToResult(routine).status, "ok");
  assert.equal(mapJsonToResult({ ...routine, degraded: ["routine", "vector"] }).status, "degraded");
  assert.equal(mapJsonToResult({ ...routine, status: "degraded" }).status, "degraded");
  assert.equal(mapJsonToResult({ ...routine, degraded: true }).status, "degraded");
});
