import { trpc } from "@/lib/trpc";
import { UNAUTHED_ERR_MSG } from '@shared/const';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { httpBatchLink, TRPCClientError } from "@trpc/client";
import { createRoot, Root } from "react-dom/client";
import superjson from "superjson";
import App from "./App";
import { getLoginUrl } from "./const";
import "./index.css";

const queryClient = new QueryClient();

const redirectToLoginIfUnauthorized = (error: unknown) => {
  if (!(error instanceof TRPCClientError)) return;
  if (typeof window === "undefined") return;

  const isUnauthorized = error.message === UNAUTHED_ERR_MSG;

  if (!isUnauthorized) return;

  window.location.href = getLoginUrl();
};

queryClient.getQueryCache().subscribe(event => {
  if (event.type === "updated" && event.action.type === "error") {
    const error = event.query.state.error;
    redirectToLoginIfUnauthorized(error);
    console.error("[API Query Error]", error);
  }
});

queryClient.getMutationCache().subscribe(event => {
  if (event.type === "updated" && event.action.type === "error") {
    const error = event.mutation.state.error;
    redirectToLoginIfUnauthorized(error);
    console.error("[API Mutation Error]", error);
  }
});

const trpcClient = trpc.createClient({
  links: [
    httpBatchLink({
      url: "/api/trpc",
      transformer: superjson,
      fetch(input, init) {
        return globalThis.fetch(input, {
          ...(init ?? {}),
          credentials: "include",
        });
      },
    }),
  ],
});

// Create or reuse a root so HMR doesn't create multiple roots and
// cause React to try removing nodes that aren't children anymore.
declare global {
  interface Window {
    __DOC_CONVERTER_ROOT__?: Root;
  }
}

const container = document.getElementById("root")!;

// Development-time safety: wrap removeChild to ignore NotFoundError thrown
// when external scripts/extensions remove nodes React expects to manage.
// Only active in Vite dev mode.
if ((import.meta as any).env?.DEV) {
  try {
    const orig = (Node.prototype as any).removeChild;
    (Node.prototype as any).removeChild = function (child: Node) {
      try {
        return orig.call(this, child);
      } catch (err: any) {
        // Suppress DOM NotFoundError caused by external DOM mutations
        if (err && (err.name === "NotFoundError" || /Failed to execute 'removeChild'/.test(err.message || ""))) {
          console.debug("Suppressed NotFoundError in removeChild (external mutation)");
          return child;
        }
        throw err;
      }
    };
  } catch (e) {
    // If patching fails, silently continue — this is only a best-effort dev safeguard
  }
}
if (!window.__DOC_CONVERTER_ROOT__) {
  window.__DOC_CONVERTER_ROOT__ = createRoot(container);
}

window.__DOC_CONVERTER_ROOT__.render(
  <trpc.Provider client={trpcClient} queryClient={queryClient}>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </trpc.Provider>
);

// Optional HMR cleanup: if the module is disposed, unmount the root
if (import.meta && (import.meta as any).hot) {
  (import.meta as any).hot.dispose(() => {
    try {
      window.__DOC_CONVERTER_ROOT__?.unmount();
      delete window.__DOC_CONVERTER_ROOT__;
    } catch (e) {
      // ignore
    }
  });
}
