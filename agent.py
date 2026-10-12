"""
Core agent logic: call the LLM (via OpenRouter) and get back a
schema-validated StrategyDocumentAnalysis, retrying with corrective feedback
if the model returns malformed or non-conforming JSON.
"""

import json
import os
import re
import time

import openai
from openai import OpenAI
from pydantic import ValidationError

from prompts import SYSTEM_PROMPT, build_user_content
from schema import StrategyDocumentAnalysis, get_schema_for_prompt

MAX_RETRIES = 3


class AgentError(Exception):
    """Raised when the agent cannot produce valid output after all retries."""


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise AgentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _strip_code_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


MAX_TRANSIENT_RETRIES = 4  # for rate limits / transient upstream errors, separate from schema-retry logic


def _call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    """
    Calls the model, handling two distinct failure modes differently:

    1. response_format rejected outright (BadRequestError) -> some free
       models/providers don't support {"type": "json_object"}; fall back to
       a plain call once, no backoff needed.
    2. Transient errors (RateLimitError, connection errors, 5xx) -> back off
       and retry the SAME request, honoring Retry-After when the provider
       supplies one. Earlier versions of this function caught these with a
       blanket `except Exception`, which misclassified rate limits as
       "response_format unsupported" and retried immediately into the same
       limit. Free-tier OpenRouter models frequently hit shared-pool rate
       limits (429), so this distinction matters in practice, not just in
       theory.
    """
    use_json_mode = True
    last_err: Exception | None = None

    for attempt in range(1, MAX_TRANSIENT_RETRIES + 1):
        kwargs = dict(model=model, messages=messages, temperature=0.2)
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                raise AgentError("Model returned an empty response.")
            return content
        except openai.BadRequestError as e:
            if use_json_mode:
                # Likely response_format isn't supported by this model/provider.
                # Retry immediately without it -- not a transient failure.
                use_json_mode = False
                continue
            raise AgentError(f"Model rejected the request: {e}") from e
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            last_err = e
            wait_seconds = _extract_retry_after(e) or min(2 ** attempt, 20)
            print(
                f"    (transient error on attempt {attempt}/{MAX_TRANSIENT_RETRIES}: "
                f"{type(e).__name__}; retrying in {wait_seconds}s)"
            )
            time.sleep(wait_seconds)
            continue

    raise AgentError(
        f"Model call failed after {MAX_TRANSIENT_RETRIES} attempts due to transient "
        f"errors (e.g. free-tier rate limiting). Last error: {last_err}"
    ) from last_err


def _extract_retry_after(e: Exception) -> float | None:
    """Best-effort extraction of a provider-supplied Retry-After hint."""
    try:
        response = getattr(e, "response", None)
        if response is not None:
            header_val = response.headers.get("Retry-After")
            if header_val:
                return float(header_val)
    except Exception:
        pass
    return None

# Rough safeguard against exceeding the model's context window.
# It does NOT truncate or modify the document in any way: it only makes the risk visible
# instead of the request failing, timing out, or getting silently truncated
# by the provider with no explanation.
MAX_DOCUMENT_CHARS = 180_000  # ~45k tokens (~4 chars/token), roughly 65-70
                               # pages of prose. The smallest free-tier
                               # context window seen during testing was
                               # ~66k tokens; this leaves that model ~20k
                               # tokens of headroom for the system prompt,
                               # schema, warnings, images, and the model's
                               # own output, while not flagging ordinary
                               # documents that would actually process fine.


def _check_length_safeguard(document_text: str, extraction_warnings: list[str]) -> None:
    if len(document_text) <= MAX_DOCUMENT_CHARS:
        return

    extraction_warnings.append(
        f"Document text is very long ({len(document_text)} characters, "
        f"exceeding the recommended safe limit of {MAX_DOCUMENT_CHARS} "
        f"characters). The full document is still being sent, but this "
        f"may cause the request to fail, time out, or have some pages "
        f"silently ignored/truncated by the model provider -- see "
        f"WRITEUP.md for why full long-document chunking is not yet "
        f"implemented."
    )


def analyze_document(
    document_text: str,
    extraction_warnings: list[str],
    images: list[dict] | None = None,
    model: str | None = None,
) -> tuple[StrategyDocumentAnalysis, list[str]]:
    """
    Returns (validated_analysis, agent_log) where agent_log records what
    happened across retries (useful for the demo / write-up).

    `images` (optional): list of {"location", "mime_type", "base64_data"}
    dicts from parser.py. When present, the user message is sent in
    OpenAI's multimodal content format (text + image_url blocks) instead of
    a plain string, so a vision-capable model can see the actual pages
    flagged during text extraction. Requires OPENROUTER_MODEL to point at a
    model that supports image input -- see README for the model used here.
    """
    # openai/gpt-oss-20b:free (Phase 1 default) is text-only. Phase 2 needs a
    # model that supports image input too, so the default changes to
    # google/gemma-4-26b-a4b-it:free, which explicitly advertises multimodal
    # input (text/image/video) alongside structured output support. This is
    # still a free-tier model, so the usual availability caveats apply --
    # see WRITEUP.md.
    model = model or os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    client = _get_client()
    schema_json = json.dumps(get_schema_for_prompt(), indent=2)

    _check_length_safeguard(document_text, extraction_warnings)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(document_text, schema_json, extraction_warnings, images),
        },
    ]

    agent_log: list[str] = []

    for attempt in range(1, MAX_RETRIES + 1):
        agent_log.append(f"Attempt {attempt}: calling model '{model}'...")
        raw = _call_model(client, model, messages)
        cleaned = _strip_code_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            agent_log.append(f"Attempt {attempt}: JSON parse failed ({e}).")
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That response was not valid JSON (error: {e}). "
                        "Return ONLY the corrected JSON object, no other text, "
                        "no markdown code fences."
                    ),
                }
            )
            continue

        try:
            validated = StrategyDocumentAnalysis.model_validate(parsed)
            agent_log.append(f"Attempt {attempt}: validated successfully against schema.")
            return validated, agent_log
        except ValidationError as e:
            agent_log.append(f"Attempt {attempt}: schema validation failed:\n{e}")
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That JSON did not match the required schema. Validation "
                        f"errors:\n{e}\n\nReturn ONLY the corrected JSON object "
                        "that fixes these errors, no other text."
                    ),
                }
            )
            continue

    raise AgentError(
        f"Failed to get valid structured output after {MAX_RETRIES} attempts. "
        f"Log:\n" + "\n".join(agent_log)
    )
