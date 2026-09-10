import * as React from "react";

const noop = () => () => {};

/**
 * True only after the component has hydrated on the client. Implemented via
 * useSyncExternalStore (server snapshot = false, client snapshot = true)
 * instead of `useEffect(() => setMounted(true), [])` so it never triggers a
 * synchronous setState-in-effect.
 */
export function useMounted() {
  return React.useSyncExternalStore(
    noop,
    () => true,
    () => false,
  );
}
