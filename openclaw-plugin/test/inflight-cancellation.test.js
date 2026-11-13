import test from "node:test";
import assert from "node:assert/strict";
import { setImmediate as tick } from "node:timers/promises";
import { InflightDeduper, createRuntime, createBeforePromptBuildHandler } from "../index.js";

test("one canceled hook cannot cancel another identical active hook", async () => {
  const runtime = createRuntime({ cache: { enabled: false } }, { info() {}, warn() {} });
  const signals = [];
  let finish;
  runtime.backends = [{ name: "synthetic", kind: "test", configHash: "one", query({ signal }) {
    signals.push(signal);
    return new Promise((resolve) => {
      finish = resolve;
      signal.addEventListener("abort", () => resolve({ status: "error", confidence: 0, cache: { cacheable: false } }), { once: true });
    });
  }}];
  const handler = createBeforePromptBuildHandler(runtime);
  const first = new AbortController();
  const second = new AbortController();
  const context = { chatType: "direct", sessionKey: "synthetic" };
  try {
    const a = handler({ prompt: "what did we decide?", signal: first.signal }, context);
    const b = handler({ prompt: "what did we decide?", signal: second.signal }, context);
    await tick();
    assert.equal(signals.length, 1);
    first.abort(new Error("first canceled"));
    assert.equal(await a, undefined);
    assert.equal(signals[0].aborted, false);
    finish({ status: "ok", answer: "The synthetic blue connector.", confidence: 0.9 });
    assert.match((await b).prependContext, /blue connector/);
    assert.equal(runtime.inflight.size(), 0);
  } finally { await runtime.dispose(); }
});

test("all canceled subscribers cancel underlying work once", async () => {
  const deduper = new InflightDeduper();
  const a = new AbortController();
  const b = new AbortController();
  let signal;
  let aborts = 0;
  const work = (shared) => {
    signal = shared;
    return new Promise((resolve) => shared.addEventListener("abort", () => { aborts++; resolve("drained"); }, { once: true }));
  };
  const first = deduper.run("key", work, a.signal);
  const second = deduper.run("key", work, b.signal);
  const outcomes = Promise.allSettled([first.promise, second.promise]);
  await tick();
  a.abort(new Error("a"));
  assert.equal(signal.aborted, false);
  b.abort(new Error("b"));
  assert.deepEqual((await outcomes).map((r) => r.status), ["rejected", "rejected"]);
  assert.equal(aborts, 1);
  await tick();
  assert.equal(deduper.size(), 0);
});

test("already canceled subscriber never starts work", async () => {
  const deduper = new InflightDeduper();
  const controller = new AbortController();
  controller.abort(new Error("early"));
  let calls = 0;
  const result = deduper.run("key", () => { calls++; return "unexpected"; }, controller.signal);
  await assert.rejects(result.promise, /early/);
  assert.equal(calls, 0);
  assert.equal(deduper.size(), 0);
});

test("legacy callers retain shared promise identity and rejection cleanup", async () => {
  const deduper = new InflightDeduper();
  let calls = 0;
  const first = deduper.run("key", () => { calls++; throw new Error("synthetic"); });
  const second = deduper.run("key", () => { calls++; });
  assert.equal(first.promise, second.promise);
  assert.equal(second.hit, true);
  await assert.rejects(first.promise, /synthetic/);
  assert.equal(calls, 1);
  assert.equal(deduper.size(), 0);
});

test("late canceled settlement cannot remove a newer entry", async () => {
  const deduper = new InflightDeduper();
  const controller = new AbortController();
  let drainOld;
  let finishNew;
  const old = deduper.run("key", () => new Promise((resolve) => { drainOld = resolve; }), controller.signal);
  const canceled = assert.rejects(old.promise, /cancel/);
  await tick();
  controller.abort(new Error("cancel"));
  await canceled;
  const next = deduper.run("key", () => new Promise((resolve) => { finishNew = resolve; }));
  await tick();
  drainOld("drained");
  await tick();
  assert.equal(deduper.size(), 1);
  finishNew("new");
  assert.equal(await next.promise, "new");
  assert.equal(deduper.size(), 0);
});

test("a noncancelable subscriber retains the underlying request", async () => {
  const deduper = new InflightDeduper();
  const controller = new AbortController();
  let shared;
  let finish;
  const first = deduper.run("key", (signal) => { shared = signal; return new Promise((resolve) => { finish = resolve; }); });
  const second = deduper.run("key", () => assert.fail("duplicate work"), controller.signal);
  const canceled = assert.rejects(second.promise, /cancel/);
  await tick();
  controller.abort(new Error("cancel"));
  await canceled;
  assert.equal(shared.aborted, false);
  finish("done");
  assert.equal(await first.promise, "done");
  assert.equal(deduper.size(), 0);
});
