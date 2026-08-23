import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { throttleTooltip } from "../../component/src/lib/utils.js";

describe("throttleTooltip", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("keeps the default timing behavior and runs the first hover immediately", () => {
    const run = vi.fn().mockResolvedValue(undefined);
    const throttle = throttleTooltip(run, () => false);

    throttle("first");

    expect(run).toHaveBeenCalledOnce();
    expect(run).toHaveBeenCalledWith("first");
  });

  it("coalesces delayed hovers and only runs the latest pointer position", async () => {
    let visible = true;
    const run = vi.fn().mockResolvedValue(undefined);
    const throttle = throttleTooltip(run, () => visible);

    throttle("first");
    await vi.runOnlyPendingTimersAsync();
    visible = false;
    vi.advanceTimersByTime(301);

    throttle("second");
    throttle("latest");
    expect(run).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(300);
    await vi.runOnlyPendingTimersAsync();
    expect(run).toHaveBeenCalledTimes(2);
    expect(run).toHaveBeenLastCalledWith("latest");
  });

  it("uses updated timing values for a pending hover", async () => {
    let visible = true;
    let delay = 300;
    let threshold = 300;
    const run = vi.fn().mockResolvedValue(undefined);
    const throttle = throttleTooltip(run, () => visible, () => delay, () => threshold);

    throttle("first");
    await vi.runOnlyPendingTimersAsync();
    visible = false;
    vi.advanceTimersByTime(301);
    delay = 10;
    threshold = 0;

    throttle("second");
    vi.advanceTimersByTime(9);
    expect(run).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);
    await vi.runOnlyPendingTimersAsync();
    expect(run).toHaveBeenLastCalledWith("second");
  });
});
