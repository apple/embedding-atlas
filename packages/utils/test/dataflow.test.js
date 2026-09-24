// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Dataflow, DataflowValue } from "@embedding-atlas/utils";

import { describe, expect, it } from "vitest";

describe("Dataflow", () => {
  describe("DataflowValue", () => {
    it("should hold and update a value", () => {
      let v = new DataflowValue(1);
      expect(v.value).toBe(1);
      v.value = 2;
      expect(v.value).toBe(2);
    });

    it("should be created through df.value()", () => {
      let df = new Dataflow();
      let v = df.value("hello");
      expect(v).toBeInstanceOf(DataflowValue);
      expect(v.value).toBe("hello");
    });
  });

  describe("derive", () => {
    it("should compute a value from its inputs", () => {
      let df = new Dataflow();
      let a = df.value(2);
      let b = df.value(3);
      let sum = df.derive([a, b], (x, y) => x + y);
      expect(sum.value).toBe(5);
    });

    it("should accept raw (non-node) arguments", () => {
      let df = new Dataflow();
      let a = df.value(10);
      let result = df.derive([a, 100], (x, y) => x + y);
      expect(result.value).toBe(110);
    });

    it("should be lazy and memoized", () => {
      let df = new Dataflow();
      let a = df.value(2);
      let calls = 0;
      let b = df.derive([a], (x) => {
        calls += 1;
        return x * 10;
      });

      // Not evaluated until accessed.
      expect(calls).toBe(0);

      expect(b.value).toBe(20);
      expect(calls).toBe(1);

      // Repeated access returns the cached value without recomputing.
      expect(b.value).toBe(20);
      expect(calls).toBe(1);
    });

    it("should recompute when an input changes", () => {
      let df = new Dataflow();
      let a = df.value(2);
      let calls = 0;
      let b = df.derive([a], (x) => {
        calls += 1;
        return x * 10;
      });
      expect(b.value).toBe(20);
      expect(calls).toBe(1);

      a.value = 3;
      expect(b.value).toBe(30);
      expect(calls).toBe(2);
    });

    it("should not recompute when an input is set to an equal value", () => {
      let df = new Dataflow();
      let a = df.value(2);
      let calls = 0;
      let b = df.derive([a], (x) => {
        calls += 1;
        return x * 10;
      });
      expect(b.value).toBe(20);
      expect(calls).toBe(1);

      a.value = 2; // same value, no downstream invalidation
      expect(b.value).toBe(20);
      expect(calls).toBe(1);
    });

    it("should propagate changes transitively (a -> b -> c)", () => {
      let df = new Dataflow();
      let a = df.value(1);
      let b = df.derive([a], (x) => x + 1);
      let c = df.derive([b], (x) => x * 2);
      expect(c.value).toBe(4);

      a.value = 10;
      expect(c.value).toBe(22);
    });

    it("should evaluate a shared input only once per propagation (diamond)", () => {
      let df = new Dataflow();
      let a = df.value(1);
      let sharedCalls = 0;
      let shared = df.derive([a], (x) => {
        sharedCalls += 1;
        return x + 1;
      });
      let left = df.derive([shared], (x) => x * 2);
      let right = df.derive([shared], (x) => x * 3);
      let out = df.derive([left, right], (l, r) => l + r);

      // shared = 2, left = 4, right = 6, out = 10
      expect(out.value).toBe(10);
      // The shared node is computed once even though two nodes depend on it.
      expect(sharedCalls).toBe(1);

      a.value = 2;
      // shared = 3, left = 6, right = 9, out = 15
      expect(out.value).toBe(15);
      expect(sharedCalls).toBe(2);
    });
  });

  describe("statefulDerive", () => {
    it("should persist state across recomputations", () => {
      let df = new Dataflow();
      let a = df.value(1);
      let node = df.statefulDerive([a], (state, x) => {
        state.count = (state.count ?? 0) + 1;
        return [x, state.count];
      });
      expect(node.value).toEqual([1, 1]);

      a.value = 2;
      expect(node.value).toEqual([2, 2]);
    });

    it("should call state.destroy when the dataflow is destroyed", () => {
      let df = new Dataflow();
      let a = df.value(1);
      let destroyed = false;
      let node = df.statefulDerive([a], (state, x) => {
        state.destroy = () => {
          destroyed = true;
        };
        return x;
      });
      // Force evaluation so the state is initialized.
      expect(node.value).toBe(1);

      df.destroy();
      expect(destroyed).toBe(true);
    });
  });

  describe("if", () => {
    it("should select the branch based on the condition", () => {
      let df = new Dataflow();
      let cond = df.value(true);
      let result = df.if(
        cond,
        (d) => d.value("yes"),
        (d) => d.value("no"),
      );
      expect(result.value).toBe("yes");

      cond.value = false;
      expect(result.value).toBe("no");
    });

    it("should destroy the previous branch's state when the condition flips", () => {
      let df = new Dataflow();
      let cond = df.value(true);
      let destroyed = false;
      let result = df.if(
        cond,
        (d) =>
          d.statefulDerive([], (state) => {
            state.destroy = () => {
              destroyed = true;
            };
            return "yes";
          }),
        (d) => d.value("no"),
      );
      expect(result.value).toBe("yes");

      cond.value = false;
      expect(result.value).toBe("no");
      expect(destroyed).toBe(true);
    });
  });

  describe("switch", () => {
    it("should select the case matching the input", () => {
      let df = new Dataflow();
      let key = df.value("a");
      let result = df.switch(key, {
        a: (d) => d.value(1),
        b: (d) => d.value(2),
      });
      expect(result.value).toBe(1);

      key.value = "b";
      expect(result.value).toBe(2);
    });
  });

  describe("map", () => {
    it("should map each element of an array", () => {
      let df = new Dataflow();
      let items = df.value([1, 2, 3]);
      let mapped = df.map(items, (d, item) => d.derive([item], (x) => x * 2));
      expect(mapped.value).toEqual([2, 4, 6]);

      items.value = [2, 3, 4];
      expect(mapped.value).toEqual([4, 6, 8]);
    });

    it("should reuse cached entries and rebuild only new ones", () => {
      let df = new Dataflow();
      let items = df.value(["a", "b"]);
      let builds = 0;
      let mapped = df.map(items, (d, item) => {
        builds += 1;
        return d.derive([item], (x) => x.toUpperCase());
      });
      expect(mapped.value).toEqual(["A", "B"]);
      expect(builds).toBe(2);

      // "a" is reused from cache, only "c" is built.
      items.value = ["a", "c"];
      expect(mapped.value).toEqual(["A", "C"]);
      expect(builds).toBe(3);
    });

    it("should destroy the state of removed entries", () => {
      let df = new Dataflow();
      let items = df.value(["x", "y"]);
      let destroyed = [];
      let mapped = df.map(items, (d, item) =>
        d.statefulDerive([item], (state, v) => {
          state.destroy = () => destroyed.push(v);
          return v.toUpperCase();
        }),
      );
      expect(mapped.value).toEqual(["X", "Y"]);

      items.value = ["x"];
      expect(mapped.value).toEqual(["X"]);
      expect(destroyed).toEqual(["y"]);
    });
  });

  describe("subgraph and destroy", () => {
    it("should cascade destroy from a parent to its subgraphs", () => {
      let df = new Dataflow();
      let sub = df.subgraph();
      let destroyed = false;
      let node = sub.statefulDerive([], (state) => {
        state.destroy = () => {
          destroyed = true;
        };
        return 1;
      });
      expect(node.value).toBe(1);

      df.destroy();
      expect(destroyed).toBe(true);
    });

    it("should remove a destroyed subgraph from its parent", () => {
      let df = new Dataflow();
      let sub = df.subgraph();
      expect(df._children.size).toBe(1);

      sub.destroy();
      expect(df._children.size).toBe(0);
    });

    it("should not accumulate dead contexts when a condition flips repeatedly", () => {
      // The if/switch/map nodes rebuild a child Dataflow on every change. A
      // destroyed child must detach from its parent, otherwise the parent's
      // child set grows without bound over the lifetime of the graph.
      let df = new Dataflow();
      let cond = df.value(true);
      let result = df.if(
        cond,
        (d) => d.value("yes"),
        (d) => d.value("no"),
      );

      for (let i = 0; i < 100; i++) {
        cond.value = i % 2 === 0;
        // Pull the value to force the IfNode to rebuild its branch context.
        expect(result.value).toBe(i % 2 === 0 ? "yes" : "no");
      }

      // The IfNode keeps a single live branch context at any time; the
      // previous ones must have been detached rather than leaked.
      expect(df._children.size).toBeLessThanOrEqual(1);
    });
  });

  describe("assertNotNull", () => {
    it("should return the same node", () => {
      let df = new Dataflow();
      let a = df.value(5);
      expect(df.assertNotNull(a)).toBe(a);
    });
  });
});
