import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { throttleTooltip } from '../../component/src/lib/utils.js';

describe('throttleTooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('runs immediately on first call (lastVisible is undefined)', async () => {
    const func = vi.fn().mockResolvedValue('ok');
    const isVisible = vi.fn().mockReturnValue(false);
    const throttle = throttleTooltip(func, isVisible, () => 500, () => 500);

    throttle('test1');
    expect(func).toHaveBeenCalledTimes(1);
    expect(func).toHaveBeenCalledWith('test1');
  });

  it('delays execution if tooltip was shown but not recently', async () => {
    const func = vi.fn().mockResolvedValue('ok');
    let visible = true;
    const isVisible = vi.fn(() => visible);
    
    // Config: 500ms delay, 500ms threshold
    const throttle = throttleTooltip(func, isVisible, () => 500, () => 500);

    // First call, runs immediately and sets lastVisible to now
    throttle('test1');
    expect(func).toHaveBeenCalledTimes(1);
    await Promise.resolve(); // Flush microtask so running becomes false

    // Wait 600ms. now - lastVisible = 600 > 500
    vi.advanceTimersByTime(600);
    visible = false; // not visible anymore, so lastVisible doesn't update on next perform

    // Next call, should be delayed because it's after the threshold
    throttle('test2');
    expect(func).toHaveBeenCalledTimes(1); // Still 1

    // If another call comes in during delay, it replaces the pending one
    vi.advanceTimersByTime(300); // Wait 300ms
    throttle('test3'); // Reset the 500ms delay

    vi.advanceTimersByTime(300);
    expect(func).toHaveBeenCalledTimes(1); // Still 1 because delay was reset

    // Wait remaining 200ms
    vi.advanceTimersByTime(200);
    expect(func).toHaveBeenCalledTimes(2);
    expect(func).toHaveBeenCalledWith('test3');
    await Promise.resolve(); // Flush microtasks
  });

  it('runs immediately if tooltip was shown recently', async () => {
    const func = vi.fn().mockResolvedValue('ok');
    const isVisible = vi.fn().mockReturnValue(true);
    
    // Config: 500ms delay, 500ms threshold
    const throttle = throttleTooltip(func, isVisible, () => 500, () => 500);

    // First call sets lastVisible
    throttle('test1');
    expect(func).toHaveBeenCalledTimes(1);
    await Promise.resolve(); // flush microtask so running becomes false

    // Wait 200ms (less than 500ms threshold)
    vi.advanceTimersByTime(200);

    // Next call runs immediately
    throttle('test2');
    expect(func).toHaveBeenCalledTimes(2);
    expect(func).toHaveBeenCalledWith('test2');
  });
});

