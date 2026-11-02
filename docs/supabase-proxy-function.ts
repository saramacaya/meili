// Supabase Edge Function: meili data proxy (paste into the existing proxy function).
//
// Render's shared IPs are blocked by the Overpass servers and rate-limited by
// Nominatim, so the backend (backend/outbound.py, MEILI_PROXY_URL) sends
// those requests here. Supabase's servers can reach Overpass; the old version
// of this function failed only because it sent no User-Agent, which
// overpass-api.de now rejects with HTTP 406.
//
// Contract: POST {url, method, form} with header x-meili-secret.
// Needs the secret MEILI_PROXY_SECRET (Edge Functions -> Secrets), same value
// as on Render. Only the hosts below are forwarded, so it isn't an open proxy.

const ALLOWED_HOSTS = new Set([
  "overpass-api.de",
  "overpass.private.coffee",
  "maps.mail.ru",
  "overpass.kumi.systems",
  "api.gdeltproject.org",
  "nominatim.openstreetmap.org",
]);
const USER_AGENT = "Meili safety-routing app (contact: github.com/saramacaya/meili)";
const TIMEOUT_MS = 90_000;

function sameSecret(given: string | null, expected: string): boolean {
  if (!given || given.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) diff |= given.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}

const json = (body: unknown, status: number) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

Deno.serve(async (request) => {
  if (request.method !== "POST") return json({ error: "POST only" }, 405);
  const secret = Deno.env.get("MEILI_PROXY_SECRET");
  if (!secret) return json({ error: "proxy not configured" }, 503);
  if (!sameSecret(request.headers.get("x-meili-secret"), secret)) return json({ error: "forbidden" }, 403);

  let body: { url?: unknown; method?: unknown; form?: unknown };
  try { body = await request.json(); } catch { return json({ error: "invalid JSON" }, 400); }
  let target: URL;
  try { target = new URL(String(body.url)); } catch { return json({ error: "invalid url" }, 400); }
  if (target.protocol !== "https:" || !ALLOWED_HOSTS.has(target.hostname)) return json({ error: "host not allowed" }, 400);

  const form = body.form && typeof body.form === "object" ? body.form as Record<string, unknown> : null;
  const init: RequestInit = {
    method: body.method === "POST" || form ? "POST" : "GET",
    headers: { "User-Agent": USER_AGENT, Accept: "application/json, text/plain, */*" },
    signal: AbortSignal.timeout(TIMEOUT_MS),
  };
  if (form) init.body = new URLSearchParams(Object.entries(form).map(([k, v]) => [k, String(v)]));

  try {
    const upstream = await fetch(target, init);
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/octet-stream" },
    });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === "TimeoutError";
    return json({ error: timedOut ? "upstream timeout" : "upstream unreachable" }, timedOut ? 504 : 502);
  }
});
