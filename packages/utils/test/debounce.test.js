// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { debounce } from "@embedding-atlas/utils";

import { beforeEach, describe, expect, it, vi } from "vitest";

describe("debounce", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it("should debounce function calls", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    debounced();
    debounced();
    debounced();

    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledTimes(1);
  });

  it("should call function with latest arguments", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    debounced("first");
    debounced("second");
    debounced("third");

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledTimes(1);
    expect(func).toHaveBeenCalledWith("third");
  });

  it("should reset timer on each call", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    debounced();
    vi.advanceTimersByTime(50);
    debounced();
    vi.advanceTimersByTime(50);

    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(50);

    expect(func).toHaveBeenCalledTimes(1);
  });

  it("should use default time of 1000ms", () => {
    const func = vi.fn();
    const debounced = debounce(func);

    debounced();

    vi.advanceTimersByTime(999);
    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(func).toHaveBeenCalledTimes(1);
  });

  it("should handle multiple arguments", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    debounced("arg1", "arg2", "arg3");

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledWith("arg1", "arg2", "arg3");
  });

  it("should handle no arguments", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    debounced();

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledWith();
    expect(func).toHaveBeenCalledTimes(1);
  });

  it("should allow multiple separate executions", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    debounced("first");
    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledTimes(1);
    expect(func).toHaveBeenCalledWith("first");

    debounced("second");
    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledTimes(2);
    expect(func).toHaveBeenCalledWith("second");
  });

  it("should handle rapid successive calls", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    for (let i = 0; i < 10; i++) {
      debounced(i);
      vi.advanceTimersByTime(50);
    }

    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledTimes(1);
    expect(func).toHaveBeenCalledWith(9);
  });

  it("should work with different debounce times", () => {
    const func = vi.fn();
    const debounced50 = debounce(func, 50);
    const debounced200 = debounce(func, 200);

    debounced50("fast");
    debounced200("slow");

    vi.advanceTimersByTime(50);
    expect(func).toHaveBeenCalledTimes(1);
    expect(func).toHaveBeenCalledWith("fast");

    vi.advanceTimersByTime(150);
    expect(func).toHaveBeenCalledTimes(2);
    expect(func).toHaveBeenCalledWith("slow");
  });

  it("should handle complex argument types", () => {
    const func = vi.fn();
    const debounced = debounce(func, 100);

    const obj = { key: "value" };
    const arr = [1, 2, 3];
    debounced(obj, arr, 42, "string");

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledWith(obj, arr, 42, "string");
  });

  it("should handle zero delay", () => {
    const func = vi.fn();
    const debounced = debounce(func, 0);

    debounced();

    expect(func).not.toHaveBeenCalled();

    vi.advanceTimersByTime(0);

    expect(func).toHaveBeenCalledTimes(1);
  });

  it("should create independent debounced functions", () => {
    const func = vi.fn();
    const debounced1 = debounce(func, 100);
    const debounced2 = debounce(func, 100);

    debounced1("first");
    debounced2("second");

    vi.advanceTimersByTime(100);

    expect(func).toHaveBeenCalledTimes(2);
    expect(func).toHaveBeenNthCalledWith(1, "first");
    expect(func).toHaveBeenNthCalledWith(2, "second");
  });
});
