// Cloudflare Worker for the Flosswork review tool.
//   POST /api/generate  -> writes one review with a free LLM (key stays here, never in the page)
//   GET  /api/health    -> tells you whether the AI key is set and the AI is answering
// Everything else is served from /public by Cloudflare.
//
// Any OpenAI-compatible provider works. Defaults to Groq's free tier: openai/gpt-oss-120b, with llama-3.3-70b-versatile as backup.
// Change with the MODEL / FALLBACK_MODEL / AI_BASE_URL variables and the GROQ_API_KEY (or AI_API_KEY) secret.

const hits = new Map(); // small per-instance rate limiter

// ---- variety: every review gets a different combination of these directions, so the same picks never give the same text ----
const STYLES = ['warm and heartfelt', 'short and crisp', 'conversational, like telling a friend', 'specific and detail-oriented',
  'calm and appreciative', 'upbeat but natural', 'plain-spoken and sincere', 'thoughtful and measured', 'friendly and light',
  'matter-of-fact with a warm finish', 'understated and genuine', 'enthusiastic without exclamation marks'];
const LENGTHS = ['1 or 2 short sentences (about 20 to 30 words)', 'exactly 2 sentences (about 30 to 40 words)', '3 sentences (about 50 to 60 words)',
  '3 to 4 sentences (about 60 to 80 words)', '4 sentences (about 70 to 90 words)'];
const STARTS = ['begin with how the patient felt', 'begin with the result or outcome', 'begin with the treatment', 'begin with the clinic atmosphere',
  'begin with the doctor', 'begin with one specific thing the patient liked', 'begin by speaking to future patients ("If you are thinking about...")',
  'begin with the clinic name', 'begin with a short plain statement of what was done', 'begin mid-thought with what stood out most'];
const RHYTHMS = ['mix one short sentence with longer ones', 'use medium-length sentences throughout', 'one long flowing sentence, then a short one', 'mostly short, simple sentences'];
const CLOSINGS = ['end with a simple thank-you', 'end with a plain recommendation to others, without the word highly', 'end on the most important thing they liked',
  'end without any sign-off or recommendation', 'end with one short warm sentence about how the visit felt'];
const AVOID_OPENERS = ['I recently', 'I had', 'My experience', 'I visited', 'I went to', 'Thank you', 'Choosing', 'Big thanks', 'If you are', 'Dr.'];
const FLAVOR = ['honestly', 'overall', 'simply', 'genuinely', 'truly', 'especially', 'particularly', 'clearly'];
const SIM_MAX = 0.35;        // 3-word-phrase overlap above this = "too similar to a recent review"
const KEEP_RECENT = 40;

const pick = a => a[Math.floor(Math.random() * a.length)];
function sample(a, n) { const b = [...a], out = []; while (out.length < n && b.length) out.push(b.splice(Math.floor(Math.random() * b.length), 1)[0]); return out; }
function makeVariant(liked) {
  if (!liked.length) {
    return { tone: pick(STYLES), length: pick(LENGTHS.slice(0, 3)), start: pick(STARTS.filter(s => !/liked|stood out/.test(s))), rhythm: pick(RHYTHMS), closing: pick(CLOSINGS),
             focus: 'keep it general: no specific qualities, only what they came for, who looked after them and a simple thank-you', words: sample(FLAVOR, 2), avoid: sample(AVOID_OPENERS, 3) };
  }
  const lead = pick(liked);
  const focus = liked.length > 1
    ? pick([`make "${lead}" the main point and fold the other points into one short mention`,
            `mention every selected point, starting with "${lead}"`,
            `mention every selected point, with "${lead}" coming last`])
    : 'mention the selected point naturally';
  return { tone: pick(STYLES), length: pick(LENGTHS), start: pick(STARTS), rhythm: pick(RHYTHMS), closing: pick(CLOSINGS),
           focus, words: sample(FLAVOR, 2), avoid: sample(AVOID_OPENERS, 3) };
}

// ---- recent reviews: used to steer away from repeats and to catch near-duplicates ----
let RECENT = [];   // [{ t: text, open: first five words }]  (kept in memory; also in KV if a KV namespace named RECENT is bound)
const norm = s => s.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
function grams(s, n = 3) { const w = norm(s), set = new Set(); for (let i = 0; i + n <= w.length; i++) set.add(w.slice(i, i + n).join(' ')); return set; }
function similarity(a, b) {
  const A = grams(a), B = grams(b); if (!A.size || !B.size) return 0;
  let inter = 0; for (const g of A) if (B.has(g)) inter++;
  return inter / (A.size + B.size - inter);
}
function mostSimilar(text, list) { let best = { s: 0, t: '' }; for (const r of list) { const s = similarity(text, r.t); if (s > best.s) best = { s, t: r.t }; } return best; }
async function loadRecent(env) {
  if (env.RECENT) { try { const v = await env.RECENT.get('recent'); if (v) { const a = JSON.parse(v); if (Array.isArray(a)) RECENT = a; } } catch { /* keep memory copy */ } }
  return RECENT;
}
function saveRecent(env, ctx, text) {
  RECENT.push({ t: text.slice(0, 500), open: text.split(/\s+/).slice(0, 5).join(' ') });
  if (RECENT.length > KEEP_RECENT) RECENT = RECENT.slice(-KEEP_RECENT);
  if (env.RECENT) { const p = env.RECENT.put('recent', JSON.stringify(RECENT)).catch(() => {}); if (ctx && ctx.waitUntil) ctx.waitUntil(p); }
}
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
  async fetch(req, env, ctx) {
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
      if (!treatments.length) return json({ error: 'missing fields' }, 400);      // picking what they liked is optional

      const docLine = !doctors.length
        ? 'Refer to "the doctors and team" without naming anyone.'
        : `Mention these people exactly as written and in exactly this order, each once: ${doctors.map(d => `"${d}"`).join(', then ')}.` +
          (staff.length ? ` "${staff.join('", "')}" is a team member: always name her AFTER the doctor(s) before her, e.g. "along with ${staff[0]}".` : '') +
          (isTeam ? ' Also thank "the whole team".' : '');

      const system =
`You write short Google reviews for a dental clinic on behalf of a real patient, using ONLY the facts they selected.
Rules:
- First person, sounds like an ordinary patient, not marketing copy.${child ? ' The visit was for the patient\'s child: write from the parent\'s point of view and say "my child".' : ''}
- Use only the facts provided.${liked.length ? '' : ' The patient did not pick any specific points, so keep the review general: say what they came for and who looked after them, with a simple thank-you. Do NOT name any specific quality such as comfort, hygiene, price, speed or technology.'} Do NOT invent details: no prices, numbers, timelines, before/after claims, medical claims, guarantees, staff names, or things not listed.
- Mention the clinic name "Flosswork Dental Clinic" (or just "Flosswork") once, naturally.
- ${docLine}
- ${seo.length ? `Work in this exact phrase once, naturally, inside a sentence: "${seo[0]}". Do not mention the city anywhere else.` : 'Do not mention any city or location.'}
- If several treatments are listed, mention them naturally together; do not list them like a menu.
- No emojis, hashtags, quotation marks, bullet points, or headings. No "highly recommend" cliches, no "five stars".
- Do not say you are an AI. Output ONLY the review text.`;

      const recent = await loadRecent(env);
      const v = makeVariant(liked);
      const avoid = [...new Set([...v.avoid, ...recent.slice(-8).map(r => r.open)])];
      const user =
`Treatments: ${treatments.join('; ')}
What the patient liked: ${liked.length ? liked.join('; ') : '(nothing specific selected)'}
${extra ? `Patient's own note (weave it in faithfully): ${extra}\n` : ''}Language: natural, simple Indian English
Tone: ${v.tone}
Length: ${v.length}
Opening: ${v.start}
Sentence rhythm: ${v.rhythm}
Closing: ${v.closing}
Focus: ${v.focus}
If they fit naturally, you may use each of these words once: ${v.words.join(', ')}
Do not start the review with any of these: ${avoid.map(a => `"${a}"`).join(', ')}
Variation code (make this review different from any other): ${clean(b.seed, 20)}`;

      const startedAt = Date.now();
      const tidy = t => t.replace(/^["'“”\s]+|["'“”\s]+$/g, '').replace(/^(review|here'?s.*?):\s*/i, '').trim();
      const r = await llm(env, [{ role: 'system', content: system }, { role: 'user', content: user }]);
      if (!r.ok) return json({ error: r.error, fatal: !!r.fatal }, r.keyConfigured === false ? 500 : 502);

      let text = tidy(r.text), model = r.model, fellBack = !!r.fellBack;
      if (text.length < 20) return json({ error: 'the AI returned an empty answer' }, 502);

      // too close to a recent review? ask once for a completely different version (only if there is time left)
      let best = mostSimilar(text, recent), regenerated = false;
      if (best.s >= SIM_MAX && Date.now() - startedAt < 4500) {
        const retryUser = user + `\nImportant: a first draft was too close to an existing review. Write a completely different version: a different opening, a different sentence order and different wording. Do not copy phrases from this existing review:\n"${best.t}"`;
        const r2 = await llm(env, [{ role: 'system', content: system }, { role: 'user', content: retryUser }]);
        if (r2.ok) {
          const t2 = tidy(r2.text);
          if (t2.length >= 20) { const s2 = mostSimilar(t2, recent).s; regenerated = true; if (s2 < best.s) { text = t2; best = { s: s2, t: best.t }; model = r2.model; fellBack = fellBack || !!r2.fellBack; } }
        }
      }
      saveRecent(env, ctx, text);
      return json({ review: text, model, ms: Date.now() - startedAt, fellBack, similarity: Number(best.s.toFixed(2)), regenerated,
                    variant: { tone: v.tone, length: v.length, opening: v.start, rhythm: v.rhythm, closing: v.closing } });
    }

    return env.ASSETS.fetch(req);
  }
};
