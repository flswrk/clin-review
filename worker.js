// Cloudflare Worker for the Flosswork review tool.
//   POST /api/generate  -> writes one review with a free LLM (key stays here, never in the page)
//   GET  /api/health    -> tells you whether the AI key is set and the AI is answering
// Everything else is served from /public by Cloudflare.
//
// Any OpenAI-compatible provider works. Defaults to Groq's free tier: openai/gpt-oss-120b, with llama-3.3-70b-versatile as backup.
// Change with the MODEL / FALLBACK_MODEL / AI_BASE_URL variables and the GROQ_API_KEY (or AI_API_KEY) secret.

const hits = new Map(); // small per-instance rate limiter

const STYLES = ['warm and heartfelt', 'short and crisp', 'conversational, like telling a friend',
  'specific and detail-oriented', 'calm and appreciative', 'upbeat but natural'];
const LENGTHS = ['exactly 2 sentences (about 30 to 40 words)', '3 sentences (about 50 to 60 words)',
  '3 to 4 sentences (about 60 to 80 words)'];
const STARTS = ['begin with how the patient felt', 'begin with the result or outcome',
  'begin with the treatment', 'begin with the clinic atmosphere', 'begin with the doctor'];

const pick = a => a[Math.floor(Math.random() * a.length)];
const clean = (s, n) => String(s ?? '').replace(/[\u0000-\u001f]/g, ' ').trim().slice(0, n);
const json = (o, status = 200) => new Response(JSON.stringify(o),
  { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });

function limited(ip, bucket, max) {
  const k = bucket + ip, now = Date.now();
  const recent = (hits.get(k) || []).filter(t => now - t < 60000);
  if (recent.length >= max) return true;
  recent.push(now); hits.set(k, recent);
  return false;
}

const isReasoning = m => /^openai\/gpt-oss/.test(m);

// One call to one model. Reasoning models (GPT-OSS) need a bigger token budget because their thinking
// counts against it; we keep the thinking short (low) and out of the reply.
async function callModel(base, key, model, messages, { max_tokens, temperature, timeoutMs }) {
  const body = { model, temperature, top_p: 0.95, messages };
  if (isReasoning(model)) { body.max_completion_tokens = 900; body.reasoning_effort = 'low'; body.include_reasoning = false; }
  else body.max_tokens = max_tokens;
  const t0 = Date.now();
  try {
    const r = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs)
    });
    const ms = Date.now() - t0;
    if (!r.ok) {
      let detail = '';
      try { detail = (await r.json())?.error?.message || ''; } catch { /* ignore */ }
      return { ok: false, model, ms, status: r.status, fatal: [401, 403].includes(r.status),
               error: `${model}: provider returned ${r.status}${detail ? ' (' + String(detail).slice(0, 100) + ')' : ''}` };
    }
    const d = await r.json();
    const choice = d?.choices?.[0];
    const text = (choice?.message?.content || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    if (!text) return { ok: false, model, ms, error: `${model}: empty answer${choice?.finish_reason === 'length' ? ' (ran out of tokens)' : ''}` };
    return { ok: true, model, ms, text };
  } catch (e) {
    return { ok: false, model, ms: Date.now() - t0,
             error: e.name === 'TimeoutError' ? `${model}: timed out` : `${model}: could not reach the AI provider` };
  }
}

// Main model first; if it fails (rate limit, outage, retired model, empty answer) try the backup model.
// Groq rate-limits per model, so the backup also adds capacity. A bad key is not retried.
async function llm(env, messages, opts = {}) {
  const base = env.AI_BASE_URL || 'https://api.groq.com/openai/v1';
  const primary = env.MODEL || 'openai/gpt-oss-120b';
  const backup = env.FALLBACK_MODEL === 'none' ? null : (env.FALLBACK_MODEL || 'llama-3.3-70b-versatile');
  const key = env.GROQ_API_KEY || env.AI_API_KEY;
  if (!key) return { ok: false, keyConfigured: false, fatal: true, error: 'no API key configured on the server', base, model: primary };
  const o = { max_tokens: 260, temperature: 1, ...opts };

  const r1 = await callModel(base, key, primary, messages, { ...o, timeoutMs: o.timeoutMs ?? 5000 });
  if (r1.ok) return { ...r1, keyConfigured: true, base };
  if (r1.fatal || !backup || backup === primary) return { ...r1, keyConfigured: true, base };

  const r2 = await callModel(base, key, backup, messages, { ...o, timeoutMs: 4500 });
  if (r2.ok) return { ...r2, keyConfigured: true, base, fellBack: true, primaryError: r1.error };
  return { ...r2, keyConfigured: true, base, error: `${r1.error}; backup ${r2.error}` };
}

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const ip = req.headers.get('CF-Connecting-IP') || 'x';

    /* ---- health check ---- */
    if (url.pathname === '/api/health') {
      if (limited(ip, 'h', 6)) return json({ ok: false, error: 'too many checks, wait a minute' }, 429);
      const r = await llm(env, [{ role: 'user', content: 'Reply with the single word OK.' }], { max_tokens: 8, temperature: 0 });
      const answered = r.ok && r.text.length > 0;
      return json({
        ok: answered, keyConfigured: r.keyConfigured, ms: r.ms ?? 0,
        provider: new URL(r.base).host, model: r.model,
        note: answered && r.fellBack ? `Main model is failing (${r.primaryError}). The backup model is answering.` : null,
        error: answered ? null : (r.error || 'the AI returned an empty answer')
      });
    }

    /* ---- review generation ---- */
    if (url.pathname === '/api/generate') {
      if (req.method !== 'POST') return json({ error: 'method not allowed' }, 405);
      if (limited(ip, 'g', 10)) return json({ error: 'too many requests' }, 429);

      let b;
      try { b = await req.json(); } catch { return json({ error: 'bad request' }, 400); }

      const doctors = Array.isArray(b.doctors) ? b.doctors.slice(0, 5).map(x => clean(x, 60)).filter(Boolean) : [];
      const staff = Array.isArray(b.staffNames) ? b.staffNames.slice(0, 3).map(x => clean(x, 60)).filter(Boolean) : [];
      const treatments = Array.isArray(b.treatments) ? b.treatments.slice(0, 3).map(x => clean(x, 60)).filter(Boolean) : [];
      const liked = Array.isArray(b.liked) ? b.liked.slice(0, 6).map(x => clean(x, 50)).filter(Boolean) : [];
      const seo = Array.isArray(b.seo) ? b.seo.slice(0, 1).map(x => clean(x, 60)).filter(Boolean) : [];
      const extra = clean(b.extra, 200);
      const isTeam = !!b.isTeam, child = !!b.child;
      if (!treatments.length || !liked.length) return json({ error: 'missing fields' }, 400);

      const docLine = !doctors.length
        ? 'Refer to "the doctors and team" without naming anyone.'
        : `Mention these people exactly as written and in exactly this order, each once: ${doctors.map(d => `"${d}"`).join(', then ')}.` +
          (staff.length ? ` "${staff.join('", "')}" is a team member: always name her AFTER the doctor(s) before her, e.g. "along with ${staff[0]}".` : '') +
          (isTeam ? ' Also thank "the whole team".' : '');

      const system =
`You write short Google reviews for a dental clinic on behalf of a real patient, using ONLY the facts they selected.
Rules:
- First person, sounds like an ordinary patient, not marketing copy.${child ? ' The visit was for the patient\'s child: write from the parent\'s point of view and say "my child".' : ''}
- Use only the facts provided. Do NOT invent details: no prices, numbers, timelines, before/after claims, medical claims, guarantees, staff names, or things not listed.
- Mention the clinic name "Flosswork Dental Clinic" (or just "Flosswork") once, naturally.
- ${docLine}
- ${seo.length ? `Work in this exact phrase once, naturally, inside a sentence: "${seo[0]}". Do not mention the city anywhere else.` : 'Do not mention any city or location.'}
- If several treatments are listed, mention them naturally together; do not list them like a menu.
- No emojis, hashtags, quotation marks, bullet points, or headings. No "highly recommend" cliches, no "five stars".
- Do not say you are an AI. Output ONLY the review text.`;

      const user =
`Treatments: ${treatments.join('; ')}
What the patient liked: ${liked.join('; ')}
${extra ? `Patient's own note (weave it in faithfully): ${extra}\n` : ''}Language: natural, simple Indian English
Tone: ${pick(STYLES)}
Length: ${pick(LENGTHS)}
Structure: ${pick(STARTS)}
Variation code (make this review different from any other): ${clean(b.seed, 20)}`;

      const r = await llm(env, [{ role: 'system', content: system }, { role: 'user', content: user }]);
      if (!r.ok) return json({ error: r.error, fatal: !!r.fatal }, r.keyConfigured === false ? 500 : 502);

      const text = r.text.replace(/^["'“”\s]+|["'“”\s]+$/g, '').replace(/^(review|here'?s.*?):\s*/i, '').trim();
      if (text.length < 20) return json({ error: 'the AI returned an empty answer' }, 502);
      return json({ review: text, model: r.model, ms: r.ms, fellBack: !!r.fellBack });
    }

    return env.ASSETS.fetch(req);
  }
};
