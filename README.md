# Strategy Document Analysis Agent 

Analyzes a strategy/business document (PDF or DOCX) and returns a
structured, schema-validated, consulting-style JSON assessment.

## Architecture

Five files, each with one job:

- `main.py` — CLI entry point; wires everything together.
- `parser.py` — extracts text from PDF/DOCX.
- `prompts.py` — the system prompt and user-prompt builder.
- `schema.py` — the Pydantic models defining the required JSON output.
- `agent.py` — calls the LLM, validates its output, retries on failure.

**Flow:** `parser.extract_text()` returns the document's text plus a list of
extraction warnings (e.g. "page 5 contains an embedded image not
analyzed"). `prompts.build_user_prompt()` combines that text, the JSON
schema, and the warnings into the message sent to the model. `agent.analyze_document()`
calls the model, parses its response as JSON, and validates it against the
schema, retrying with the specific error fed back to the model if
validation fails. Only a schema-valid result is ever returned or written to disk.

### Document parsing (`parser.py`)

PDFs are processed page by page with `pdfplumber`: text and tables are
extracted separately (tables labeled `[Table N on page X]`, since plain
text extraction tends to scramble columns), and pages with no text or with
embedded images are flagged as warnings. DOCX files are walked in original
element order with `python-docx`, so tables stay linked to the heading they
appeared under, and headings are marked inline (`[--- SECTION: ... ---]`)
as citation anchors.

### Schema (`schema.py`)

The output is a Pydantic model (`StrategyDocumentAnalysis`) covering all
fields required. Every finding, recommendation, metric, and
risk carries its own `evidence` (a source reference + a high/medium/low
confidence level), rather than only one global evidence block. 
A `processing_metadata` field separately reports parsing issues
(garbled tables, undetected images), so those don't get mixed into the
business analysis itself.

### System prompt (`prompts.py`)

Instructs the model to act as a strategy consultant, ground every claim
with a source reference and confidence level, never invent numbers, keep
the document's own stated assumptions separate from the model's own
inferences, and flag content it can't fully interpret instead of ignoring it.

### Agent logic (`agent.py`)

Two retry loops. `_call_model()` (network layer, 4 attempts) handles rate
limits and connection/server errors with backoff, and falls back once if
the provider rejects the `response_format` parameter. The outer loop
(content layer, 3 attempts) handles JSON-parsing and schema-validation
failures by feeding the specific error back to the model before retrying.

## Setup

```bash
cd strategy-agent
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

Note: OpenRouter's free-tier lineup changes over time and models can get
delisted without notice (see `WRITEUP.md`). If you get a 404 on the
configured model, check https://openrouter.ai/collections/free-models for
a current `:free` slug.
 
## Running it

```bash
python main.py test_documents/market_entry_study.docx --verbose
python main.py test_documents/transformation_roadmap.pdf --out outputs/roadmap.json
```

Output is written to `outputs/<filename>.json` by default. `--verbose`
prints the agent's attempt log (each retry and why).
