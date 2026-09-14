"""
Web interface for the dual-pass RAG chatbot.

A thin layer over controller.answer_query(). It performs no routing, no
verification and no fallback logic of its own: everything it displays is
produced by the same code path the evaluation used, so the demonstration
shows the system that was measured rather than a re-implementation of it.

The interface shows the answer alone by default, which is what a user of a
deployed system would see. A details panel exposes the machinery beneath —
the outcome, the retrieved passages, the claim-by-claim verdicts, and the
timings — because the point of the demonstration is that this layer exists
and can be inspected.

The active configuration is fixed by ACTIVE_ARM before the server starts and
is not selectable in the interface, matching the evaluation setup.

Run:
    uv run uvicorn web.main:app --reload
    ACTIVE_ARM=baseline uv run uvicorn web.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from pathlib import Path

from src.config import (
    ACTIVE_ARM,
    ACTIVE_NODE_B_MODEL,
    CORPUS_DIR,
    NODE_A_MODEL,
    NODE_B_ENABLED,
)
from src.controller import answer_query

app = FastAPI(title="Verified FAQ assistant")

_corpus_cache: dict[str, str] = {}


def _passage_text(chunk_id: str) -> str:
    """The text behind a chunk identifier, so the details panel can show the
    evidence a verdict was judged against rather than only its label."""
    faq_id = chunk_id.split("::")[0]
    if faq_id not in _corpus_cache:
        path = Path(CORPUS_DIR) / f"{faq_id}.md"
        _corpus_cache[faq_id] = (
            path.read_text(encoding="utf-8").strip() if path.exists()
            else "(passage not found)")
    return _corpus_cache[faq_id]


class Question(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    verifier = ACTIVE_NODE_B_MODEL if NODE_B_ENABLED else "none"
    return PAGE.replace("{{ARM}}", ACTIVE_ARM) \
               .replace("{{GENERATOR}}", NODE_A_MODEL) \
               .replace("{{VERIFIER}}", verifier)


@app.post("/ask")
def ask(question: Question) -> JSONResponse:
    """
    One question through the full pipeline. Returns what the user sees and,
    separately, what the system did to produce it.
    """
    result = answer_query(query_id="web", question=question.text)
    record = result.to_dict()

    return JSONResponse({
        "answer": record["answer_shown"],
        "outcome": record["outcome"],
        "attempts": record["attempts"],
        "passages": [
            {"id": cid, "text": _passage_text(cid)}
            for cid in record["retrieved_chunk_ids"]
        ],
        "drafts": [
            {"stage": d["stage"], "text": d["text"]}
            for d in record["drafts"]
        ],
        "verifications": [
            {
                "stage": v["stage"],
                "claims": v["claims"],
                "all_supported": v["all_supported"],
            }
            for v in record["verifications"]
        ],
        "timing": {
            "retrieval": record["t_retrieval_s"],
            "generation": record["t_generation_s"],
            "verification": record["t_verification_s"],
            "rewrite": record["t_rewrite_s"],
            "total": record["t_total_s"],
        },
        "tokens": {
            "generator": record["tokens_a_prompt"] + record["tokens_a_completion"],
            "verifier": record["tokens_b_prompt"] + record["tokens_b_completion"],
        },
    })


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verified FAQ assistant</title>
<style>
  :root {
    --ink:        #16202b;
    --ink-soft:   #55647a;
    --paper:      #f7f6f3;
    --rule:       #d8d4cc;
    --stamp:      #1f5f4f;
    --stamp-soft: #e4efe9;
    --flag:       #9a4a2f;
    --flag-soft:  #f6e9e3;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    /* rem, not px, so a reader who has enlarged their browser default keeps it */
    font: 1rem/1.55 ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 46rem; margin: 0 auto; padding: 3rem 1.25rem 5rem; }

  header { border-bottom: 1.5px solid var(--ink); padding-bottom: .85rem; margin-bottom: 2.25rem; }
  h1 { margin: 0; font-size: 1.15rem; font-weight: 650; letter-spacing: -.01em; }
  .config {
    margin-top: .4rem; font-size: .78rem; color: var(--ink-soft);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  form { display: flex; gap: .6rem; margin-bottom: 2rem; }
  input[type=text] {
    flex: 1; padding: .7rem .85rem; font: inherit;
    border: 1.5px solid var(--rule); border-radius: 3px; background: #fff;
  }
  input[type=text]:focus { outline: 2px solid var(--stamp); outline-offset: 1px; border-color: transparent; }
  button {
    padding: .7rem 1.15rem; font: inherit; font-weight: 550; cursor: pointer;
    background: var(--ink); color: var(--paper);
    border: 1.5px solid var(--ink); border-radius: 3px;
  }
  button:hover { background: #223244; }
  button:disabled { opacity: .5; cursor: default; }

  #answer { font-size: 1.05rem; white-space: pre-wrap; }
  #answer:empty { display: none; }

  .stamp {
    display: inline-block; margin-bottom: 1rem;
    padding: .2rem .55rem; border-radius: 2px;
    font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .stamp.ok       { background: var(--stamp-soft); color: var(--stamp); }
  .stamp.withheld { background: var(--flag-soft);  color: var(--flag);  }

  #details { margin-top: 2.5rem; border-top: 1px solid var(--rule); padding-top: 1rem; }
  #details[hidden] { display: none; }
  #details > summary { cursor: pointer; font-size: .85rem; color: var(--ink-soft); font-weight: 550; }
  summary:focus-visible { outline: 2px solid var(--stamp); outline-offset: 2px; }

  .panel { margin-top: 1.25rem; font-size: .85rem; }
  .panel h2 {
    font-size: .72rem; font-weight: 650; letter-spacing: .07em; text-transform: uppercase;
    color: var(--ink-soft); margin: 1.5rem 0 .5rem;
  }
  .panel h2:first-child { margin-top: 0; }

  .claim { padding: .45rem 0; border-bottom: 1px solid var(--rule); display: flex; gap: .75rem; }
  .claim:last-child { border-bottom: 0; }
  .verdict {
    flex: none; width: 6.5rem; font-size: .68rem; font-weight: 650;
    letter-spacing: .05em; text-transform: uppercase; padding-top: .15rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .supported    { color: var(--stamp); }
  .contradicted,
  .unsupported  { color: var(--flag); }

  .draft { padding: .5rem 0; border-bottom: 1px solid var(--rule); }
  .draft:last-child { border-bottom: 0; }
  .stage {
    font-size: .68rem; letter-spacing: .05em; text-transform: uppercase;
    color: var(--ink-soft); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .meta { color: var(--ink-soft); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; }
  .meta span { margin-right: 1.25rem; white-space: nowrap; }

  .passage { border-bottom: 1px solid var(--rule); }
  .passage:last-child { border-bottom: 0; }
  .passage summary {
    padding: .45rem 0; font-size: .78rem; font-weight: 500; color: var(--ink-soft);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .passage summary:hover { color: var(--ink); }
  .passage p {
    margin: 0 0 .7rem; padding-left: 1rem;
    border-left: 2px solid var(--rule); white-space: pre-wrap;
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>Verified FAQ assistant</h1>
    <div class="config">{{ARM}} &nbsp;·&nbsp; generator {{GENERATOR}} &nbsp;·&nbsp; verifier {{VERIFIER}}</div>
  </header>

  <form id="form">
    <input type="text" id="q" placeholder="Ask about leave, timelines or procedures" autocomplete="off" required>
    <button type="submit" id="go">Ask</button>
  </form>

  <div id="stamp"></div>
  <div id="answer"></div>

  <details id="details" hidden>
    <summary>How this answer was produced</summary>
    <div class="panel" id="panel"></div>
  </details>

</div>

<script>
const form = document.getElementById('form');
const qEl = document.getElementById('q');
const go = document.getElementById('go');
const answerEl = document.getElementById('answer');
const stampEl = document.getElementById('stamp');
const details = document.getElementById('details');
const panel = document.getElementById('panel');

const OUTCOME_LABEL = {
  passed_first:       ['Verified', 'ok'],
  fixed_by_rewrite:   ['Verified after revision', 'ok'],
  fallback:           ['Answer withheld', 'withheld'],
  generation_failed:  ['Could not answer', 'withheld'],
  verifier_malformed: ['Could not answer', 'withheld'],
};

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = qEl.value.trim();
  if (!question) return;

  go.disabled = true;
  go.textContent = 'Checking';
  stampEl.innerHTML = '';
  answerEl.textContent = '';
  details.hidden = true;

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: question}),
    });
    if (!res.ok) throw new Error('Request failed');
    const data = await res.json();

    const [label, cls] = OUTCOME_LABEL[data.outcome] || [data.outcome, 'ok'];
    stampEl.innerHTML = '<span class="stamp ' + cls + '">' + label + '</span>';
    answerEl.textContent = data.answer;
    panel.innerHTML = render(data);
    details.hidden = false;
  } catch (err) {
    stampEl.innerHTML = '<span class="stamp withheld">Not available</span>';
    answerEl.textContent = 'The assistant could not be reached. Try again.';
  } finally {
    go.disabled = false;
    go.textContent = 'Ask';
  }
});

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function render(d) {
  let h = '';

  h += '<h2>Outcome</h2><div class="meta">';
  h += '<span>' + esc(d.outcome) + '</span>';
  h += '<span>revisions: ' + d.attempts + '</span></div>';

  h += '<h2>Passages retrieved</h2>';
  for (const p of d.passages) {
    h += '<details class="passage"><summary>' + esc(p.id) + '</summary><p>'
       + esc(p.text) + '</p></details>';
  }

  if (d.verifications.length) {
    h += '<h2>Claim checks</h2>';
    for (const v of d.verifications) {
      if (d.verifications.length > 1) h += '<div class="stage">' + esc(v.stage) + '</div>';
      for (const c of v.claims) {
        h += '<div class="claim"><div class="verdict ' + esc(c.verdict) + '">'
           + esc(c.verdict) + '</div><div>' + esc(c.text) + '</div></div>';
      }
    }
  } else {
    h += '<h2>Claim checks</h2><div class="meta">No verification step in this configuration.</div>';
  }

  if (d.drafts.length > 1) {
    h += '<h2>Revisions</h2>';
    for (const dr of d.drafts) {
      h += '<div class="draft"><div class="stage">' + esc(dr.stage) + '</div>'
         + esc(dr.text) + '</div>';
    }
  }

  const tVerify = d.timing.verification.reduce((a, b) => a + b, 0);
  const tRewrite = d.timing.rewrite.reduce((a, b) => a + b, 0);
  h += '<h2>Cost</h2><div class="meta">';
  h += '<span>generator ' + d.tokens.generator + ' tokens, '
     + (d.timing.generation + tRewrite).toFixed(2) + ' s</span>';
  h += '<span>verifier ' + d.tokens.verifier + ' tokens, '
     + tVerify.toFixed(2) + ' s</span>';
  h += '<span>total ' + d.timing.total.toFixed(2) + ' s</span></div>';

  return h;
}
</script>
</body>
</html>
"""