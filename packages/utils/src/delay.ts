/** Delay for the given amount of time */
export function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve, _) => {
    setTimeout(resolve, milliseconds);
  });
}
