# Phase 1: Text only

## Libraries and models considered

- **PDF extraction**: `pdfplumber`, chosen over `PyMuPDF`/`pypdf` because it
  gives per-page text *and* a separate table-extraction API. Plain-text
  extraction alone frequently scrambles table column alignment, so pulling
  tables out separately and labeling them keeps them readable for the LLM.
- **DOCX extraction**: `python-docx`, walking the document body in original
  element order (not paragraphs and tables separately) so tables stay
  positioned relative to the headings around them.
- **LLM access**: the standard `openai` Python SDK, pointed at OpenRouter's
  OpenAI-compatible endpoint (`base_url="https://openrouter.ai/api/v1"`).
  OpenRouter has no dedicated SDK, and mimicking the OpenAI API format means
  any OpenAI-compatible tooling works against it directly.
- **Model**: `openai/gpt-oss-20b:free` on OpenRouter — a free, open-weight
  model that explicitly supports structured outputs and tool use. The
  originally configured model (`meta-llama/llama-3.3-70b-instruct:free`) was
  delisted from the free tier between setup and first run — see "Problems
  encountered" below.
- **Structured output**: `response_format={"type": "json_object"}` is
  attempted first, with a fallback to a plain call if the provider rejects
  that parameter. Reliability doesn't depend on this working, though — it's
  backed up by the schema-validate-retry loop either way.

## Decisions made

- **Per-claim evidence, not one global evidence block.** Every finding,
  recommendation, metric, and risk carries its own source reference and
  confidence level. This is what actually allows distinguishing "the
  document says X" from "the model is inferring X" on a claim-by-claim
  basis, which is the core ask of the assignment.
- **A separate `processing_metadata` field**, so parsing problems (a
  garbled table, an undetected image) are reported independently of the
  business analysis, instead of being crammed into
  `missing_or_unclear_information` alongside genuine content gaps.
- **Two independent retry loops in `agent.py`** rather than one: a
  network-layer loop for transient failures (rate limits, connection
  errors) with backoff, and a separate content-layer loop for invalid JSON
  or schema-validation failures, which feeds the specific error back to the
  model rather than just asking again blindly.
- **Extraction warnings are injected into the prompt itself**, not just
  logged separately, so the model can lower its confidence on content it's
  been told may be incomplete or unreliable.

## Experiments attempted

Four test documents were built to probe specific behaviors, rather than
relying on one "easy" example:

1. **`market_entry_study.docx`** — a clean, well-structured document with
   headings that map closely onto the schema's fields. Run successfully
   end-to-end: every finding/metric/risk came back with a correct section
   reference and a "high" confidence level, and the model's
   `missing_or_unclear_information` and follow-up questions were genuinely
   useful (e.g. flagging that no cost breakdown was given). Reviewed
   claim-by-claim against the source text to confirm the citations were
   accurate, not just plausible-looking.
2. **`transformation_roadmap.pdf`** — a PDF with a financial table and an
   embedded image on page 2, to test PDF/table parsing and image detection.
   Parsing correctly flagged the embedded image as an extraction warning
   before the model was even called, confirming the detection logic works.
3. **`growth_initiative.docx`** — deliberately vague and
   internally contradictory (two different FY26 revenue figures; a
   forward-looking growth claim with no stated number), built to test
   whether the model actually uses "low" confidence and flags
   contradictions, rather than defaulting everything to "high" or silently
   picking one number.
4. **`leadership_offsite_notes_unstructured.docx`** — informal meeting
   notes with no section headings at all, findings/metrics/risks all mixed
   within the same paragraphs. Built to test whether the model is genuinely
   classifying content into the right schema categories, rather than
   succeeding by pattern-matching section titles onto field names (which is
   trivially easy on documents 1–3).

Test cases 3 and 4 are included specifically because 1 and 2 are "easy" and
well-organized documents where headings nearly match the schema. A
solution that only works on documents 3 and 4 also correctly is a
meaningfully stronger result than one that only works on 1 and 2.

## Problems encountered

- Free-tier models can reject `response_format` outright or ignore "JSON
  only" instructions and wrap output in markdown code fences — handled by
  `_strip_code_fences()` and the fallback call path in `agent.py`.
- The original error-handling in `agent.py` caught `response_format`
  failures and rate-limit errors with the same blanket `except Exception`,
  so a 429 rate limit from OpenRouter's shared free-tier pool was
  misread as "this model doesn't support `response_format`" and retried
  immediately into the same limit, causing an unhandled crash. Fixed by
  catching rate-limit/connection/server errors separately and backing off
  (honoring the provider's `Retry-After` header when present) before
  retrying, up to 4 attempts.
- The free model configured at setup time was delisted from OpenRouter's
  free tier before the project was first run, returning a 404. Free-tier
  model availability is not stable — worth re-checking before any demo or
  evaluation run.

## Known limitations

- **No visual understanding.** Charts, diagrams, and scanned/image-only
  pages are detected and flagged, but their content is not analyzed. A
  market-sizing chart with no accompanying data table currently
  contributes nothing to `business_metrics`.
- **No chunking strategy for long documents.** Very long documents may
  exceed model context limits; not handled gracefully yet.
- **Free-model variability**: rate limits, availability, and JSON-mode
  support vary between OpenRouter's free models and change over time.
- **Per-claim confidence is an unverified LLM judgment call, not a computed
  metric.** The pipeline validates that `confidence` is one of
  high/medium/low, but nothing checks whether that label is actually
  correct for the specific claim: a model could mark an inferred number as
  "high" and nothing would catch it. This is a structural limitation of
  using an LLM's self-report as a reliability signal: it guarantees the
  *shape* of the output, not the *accuracy* of the judgment inside it.
  Manually spot-checking claims against the source document (as done for
  the test cases above) is currently the only verification method.

## Improvements next

- **Visual content**: render pages flagged by
  `extraction_warnings` as images and send them to a vision-capable model
  alongside the text, rather than sending every page as an image (cheaper,
  and only invoked where the text pipeline already signaled it can't cope).
- **Second-pass confidence verification**: a separate LLM call that grades
  each claim's stated confidence against the retrieved source text
  independently, to catch cases where a claim was mislabeled "high" despite
  being an inference.

---
 
# Phase 2: Visual Content
 
## Libraries and models considered
 
- **Image extraction**: no new library was needed. `pdfplumber`'s
  `page.to_image()` (already a transitive feature via its existing
  `Pillow` dependency) renders a full screenshot of any PDF page flagged
  during Phase 1's existing image-detection check. `python-docx`'s
  existing `doc.part.rels` mechanism — already used in Phase 1 just to
  *count* embedded images — was extended to also pull out the actual image
  bytes.
- **Encoding**: Python's built-in `base64` module converts extracted image
  bytes into the text-safe `data:image/png;base64,...` format required to
  embed an image inside a JSON API request. 
- **Model**: `openai/gpt-oss-20b:free` (the Phase 1 default) is text-only
  and has no image input capability at all, so a vision-capable model was
  required. This went through three different free models over the course
  of testing, in this order:
  1. `google/gemma-4-26b-a4b-it:free` — explicitly advertises multimodal
     input (text/image/video) and structured output support. Successfully
     read a real chart's data on the first live test (see "Experiments
     attempted" below), but later became persistently rate-limited on a
     subsequent document (4/4 retry attempts exhausted).
  2. `nvidia/nemotron-nano-12b-v2-vl:free` — identified as a strong
     candidate (its description specifically calls out OCR and document-
     image understanding), but returned "No endpoints found" (404) when
     actually tried — delisted from the free tier between being noted as a
     candidate and being used.
  3. `minimax/minimax-m3:free` — a large (1M context), heavily-used free
     model with explicit image support, adopted as the more stable choice
     after the above two failures.
  This churn — three different model names tried within a single testing
  session — is direct, firsthand evidence of the free-tier model
  instability the assignment itself warns about, not just a hypothetical
  risk documented in the abstract.
## Decisions made
 
- **Pages with images, or with no readable text at all, get flagged during
  extraction: only those pages are turned into an image for the model.**
  Every other page stays on the normal, cheaper text-only path.
- **For PDFs, we screenshot the whole flagged page instead of cutting out
  just the embedded image.** A chart is often made of several separate
  layers, so a full screenshot shows everything a human would actually see.
- **Images use the same `Evidence.source_reference` field as text**, just
  with a label like "chart on page 1" instead of a section heading — no
  new schema fields needed.
- **The message sent to the model only changes format (text + images)
  when images exist**; text-only documents work exactly as before.


## Experiments attempted 
 
1. **`market_sizing_chart.pdf`**: a full strategy document (executive
   summary, strategic objective, regional performance, key risks,
   recommendation) with a real embedded bar chart showing Q4 regional
   revenue (North America $18.4$M, Europe $12.7$M, APAC $9.3$M, LATAM $4.1$M).
   **Confirmed successful end-to-end**: all four correct dollar figures
   came back in `business_metrics`, each correctly cited as `"chart on
   page 1"` with `"high"` confidence. These numbers exist nowhere in the
   document's text, the only way they could appear in the output is if
   the model genuinely read the chart. This is the strongest evidence so
   far that the vision pipeline works as designed, not just that it
   avoids hallucinating.
2. **`board_update_scanned.pdf`** : a page with zero real text layer at
   all (the text was rendered onto a flattened image, simulating a
   scanned document), a page where
   `pdfplumber.extract_text()` returns nothing and the *entire* page must
   be read as a picture. Verified offline that the page genuinely has no
   extractable text and that a legible screenshot was correctly captured.
   Live testing was disrupted by the model-availability churn described
   above (rate limits, then a delisted model) before a confirmed
   successful result was obtained.

## Problems encountered 

- Rate limiting was noticeably worse for vision-capable free models than
  for the text-only model used in Phase 1, most likely because far fewer
  providers host free vision models, so the shared capacity pool is
  smaller. This was hit directly during testing of `board_update_scanned.pdf`
  (4/4 retry attempts exhausted on `google/gemma-4-26b-a4b-it:free`), and
  the next candidate model (`nvidia/nemotron-nano-12b-v2-vl:free`) turned
  out to be delisted entirely by the time it was tried.
---
 
# Handling long documents
 
Sending the entire document in one request can exceed the model's context
window on very long documents, causing an outright failure or a silent
truncation by the API. Three approaches were considered: 

- (1) chunk the document, analyze each piece separately, then run a final 
 merge call to combine the partial results into one schema-valid output; 
- (2) summarize each chunk first, then run a single analysis pass on the 
 compressed summaries; 
- (3) true retrieval (RAG) — embed chunks and retrieve only the
 most relevant ones per field. 

Chunk-and-merge (1) is the approach I would
have implemented, since it doesn't lose information up front the way
summarizing does and doesn't require building a new retrieval subsystem.
It wasn't implemented here because it needs several LLM calls per
document instead of just one, and we were already hitting rate limits and
model outages with a single call (see "Problems encountered" above). More
calls would only have made that worse, so we couldn't build and test it
reliably in time. Instead, a small safeguard was added (`_check_length_safeguard()` in `agent.py`, called inside
`analyze_document()`) which detects when the extracted text is unusually
long and records a warning that the request may fail, time out, or have
content silently dropped, without truncating or modifying the document
itself.

**Test:** a long document (`expansion_plan.docx` 289,542 characters, well past the
threshold) was tested, and the warning appeared correctly in
`processing_metadata.extraction_warnings`: *"Document text is very long
(289542 characters, exceeding the recommended safe limit of 180000
characters). The full document was sent and processed, but this may have
caused some content to be silently truncated by the model provider;
per-claim confidence on fine details should be treated as low."*

---
 
# Making this usable by anyone 
 
Right now this only works as a command someone runs in their own terminal,
with Python and the dependencies installed locally, and their own
OpenRouter key in a local `.env` file. Turning it into something anyone
could use means adding a few layers on top of
the existing pipeline (`parser.py` + `schema.py` + `agent.py`, which would
stay as the core engine, unchanged):
 
- **Backend / API.** Wrap `main.py`'s logic behind a small web server (e.g.
  FastAPI) exposing an endpoint like `POST /analyze` that accepts an
  uploaded file and returns the JSON result over HTTP. This turns "run a
  Python script" into "make a web request," which any browser or app can
  do without needing Python installed.
- **Frontend.** A simple web page where someone uploads a file, clicks
  "Analyze," and sees the results as a readable report rather than a raw
  `.json` file. It only talks to the backend API, it doesn't run any
  analysis logic itself.
- **Hosting.** The backend needs to run continuously on a server with a
  public address, not just on your laptop while you happen to have it
  open. Deploying it to a cloud host (e.g. Render, Railway, Fly.io, or a
  major cloud provider) gives it a URL that works for anyone, anytime.
- **API key handling.** The OpenRouter key must never reach the
  frontend/browser, it stays only on the backend, which is the only thing
  that talks to OpenRouter. Either you pay for everyone's usage centrally
  (and add per-user rate limits to control cost), or each user supplies
  their own key at request time, used once and never stored.

 