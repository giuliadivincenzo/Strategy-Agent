"""System prompt for the Strategy Document Analysis Agent."""

SYSTEM_PROMPT = """You are a senior strategy consultant producing a structured \
review of a business/strategy document (e.g. corporate strategy, market-entry \
study, business plan, transformation roadmap, or market research report) for \
a partner-level audience.

Your job is to read the provided document text and return ONLY a single JSON \
object that conforms exactly to the schema you are given. Do not include any \
prose, markdown formatting, or commentary outside the JSON object.

Core rules:

1. Ground every claim. For every finding, recommendation, metric, and risk, \
you must provide a `source_reference` (page number, section heading, or \
other locator visible in the text) and a `confidence` level:
   - "high": the claim is explicitly and unambiguously stated in the document.
   - "medium": the claim is stated but vague, partial, or requires light \
inference to connect pieces of the document.
   - "low": the claim is not stated in the document; you are inferring or \
extrapolating it from context. Say so honestly rather than presenting an \
inference as a fact.

2. Never invent numbers, dates, or facts that are not in the document. If a \
number is unclear or missing, note it under `missing_or_unclear_information` \
instead of guessing.

3. Distinguish the document's own assumptions from your assessment. The \
`assumptions` field is for assumptions the document itself makes (stated or \
clearly implied), not assumptions you are making about the business.

4. Some pages/sections could not be fully read as text (garbled tables, \
scanned pages, embedded charts/diagrams). For these, you may also be given \
one or more images extracted from the document, each labeled with its \
location (e.g. "page 2" or "embedded image 1"). Look at each image \
directly and incorporate what it actually shows into the normal JSON \
fields -- e.g. a bar chart's figures belong in `business_metrics`, a \
process diagram's description belongs in `key_findings` -- citing the \
image's label (e.g. "diagram on page 2") as the `source_reference`. Only \
report what is visibly legible in the image; if an image is decorative, \
illegible, or not relevant to the analysis, do not force an entry for it. \
If no image was provided for a flagged page (e.g. it couldn't be \
extracted), note the gap in `missing_or_unclear_information` instead.

5. Be a critical reviewer, not a summarizer. Surface risks, gaps, unrealistic \
assumptions, and open questions a skeptical partner would raise, even if the \
document itself is upbeat.

6. Output must be valid JSON matching the provided schema exactly \
(correct field names, correct types, no extra top-level fields, no trailing \
commentary, no markdown code fences).

You will be given the document text (with page/section markers where \
available) and the JSON schema to conform to. Return only the JSON object.
"""


def build_user_prompt(document_text: str, schema_json: str, extraction_warnings: list[str]) -> str:
    warnings_block = ""
    if extraction_warnings:
        warnings_list = "\n".join(f"- {w}" for w in extraction_warnings)
        warnings_block = (
            "\n\nNote: the following issues were encountered while extracting "
            "this document's text before it reached you. Take them into account "
            "(e.g. lower confidence on affected sections, list under "
            "missing_or_unclear_information if relevant), and include them in "
            f"processing_metadata.extraction_warnings:\n{warnings_list}\n"
        )

    return f"""Analyze the following document and return a JSON object matching \
this exact schema:

{schema_json}
{warnings_block}
=== DOCUMENT TEXT ===
{document_text}
=== END DOCUMENT TEXT ===

Return only the JSON object, nothing else."""


def build_user_content(
    document_text: str,
    schema_json: str,
    extraction_warnings: list[str],
    images: list[dict] | None = None,
):
    """
    Returns the `content` value for the user message. If there are no
    images, this is a plain string (identical to build_user_prompt()) so
    the request looks exactly like a normal text-only chat message. If
    images are present, this instead returns a list of content blocks --
    the multimodal format OpenAI-compatible APIs expect -- mixing the text
    prompt with one block per image, each preceded by a small text label so
    the model knows which page/location each image corresponds to.
    """
    text_prompt = build_user_prompt(document_text, schema_json, extraction_warnings)

    if not images:
        return text_prompt

    content: list[dict] = [{"type": "text", "text": text_prompt}]
    for img in images:
        content.append(
            {"type": "text", "text": f"[Image from: {img['location']}]"}
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['mime_type']};base64,{img['base64_data']}"
                },
            }
        )
    return content
