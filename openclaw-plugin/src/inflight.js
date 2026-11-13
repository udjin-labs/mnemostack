/**
 * De-duplicate concurrent async work by key and clean up after settlement.
 */
export class InflightDeduper {
  constructor() {
    this.map = new Map();
  }

  run(key, fn, signal) {
    if (signal?.aborted) {
      return { promise: Promise.reject(signal.reason || new Error("aborted")), hit: this.map.has(key) };
    }
    let entry = this.map.get(key);
    const hit = Boolean(entry);
    if (!entry) {
      entry = { controller: new AbortController(), users: 0, settled: false };
      entry.promise = Promise.resolve().then(() => {
        if (entry.controller.signal.aborted) throw entry.controller.signal.reason || new Error("aborted");
        return fn(entry.controller.signal);
      }).finally(() => {
        entry.settled = true;
        if (this.map.get(key) === entry) this.map.delete(key);
      });
      this.map.set(key, entry);
    }
    entry.users++;
    // Preserve the original shared-promise API for noncancelable callers.
    if (!signal) return { promise: entry.promise, hit };
    const promise = new Promise((resolve, reject) => {
      let done = false;
      const finish = (settle, value) => {
        if (done) return;
        done = true;
        signal.removeEventListener("abort", abort);
        entry.users--;
        settle(value);
      };
      const abort = () => {
        finish(reject, signal.reason || new Error("aborted"));
        if (!entry.settled && entry.users === 0) {
          entry.controller.abort(signal.reason || new Error("aborted"));
          if (this.map.get(key) === entry) this.map.delete(key);
        }
      };
      signal.addEventListener("abort", abort, { once: true });
      entry.promise.then((value) => finish(resolve, value), (error) => finish(reject, error));
      if (signal.aborted) abort();
    });
    return { promise, hit };
  }

  size() {
    return this.map.size;
  }
}
