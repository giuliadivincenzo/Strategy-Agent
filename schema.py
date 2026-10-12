"""
Structured output schema for the Strategy Document Analysis Agent.

Design notes (see README for full reasoning):
- Every substantive claim (finding, recommendation, metric, risk) carries its
  own `evidence` block with a source reference and a confidence level, rather
  than one global "evidence" section. This is what lets the agent distinguish
  things it read directly in the document from things it's inferring, per
  claim, which is the core "evidence vs. assumption" requirement.
- `ProcessingMetadata` is an extension beyond the assignment's minimum field
  list. It exists to satisfy the "communicate errors, uncertainty, or
  unsupported content clearly" functional requirement independently of the
  business analysis itself (e.g. "page 5 contains a chart that was not
  processed in this pass").
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"       # explicitly and unambiguously stated in the document
    MEDIUM = "medium"   # stated but vague, partial, or requires light inference
    LOW = "low"         # not stated; the agent is inferring or extrapolating


class Evidence(BaseModel):
    source_reference: str = Field(
        description="Page number, section heading, an image/diagram label "
        "(e.g. 'diagram on page 2'), or other locator in the source "
        "document. Use 'not specified in document' if the claim is an "
        "inference rather than a direct citation."
    )
    confidence: ConfidenceLevel


class Finding(BaseModel):
    statement: str
    evidence: Evidence


class Recommendation(BaseModel):
    statement: str
    rationale: Optional[str] = Field(
        default=None, description="Why this recommendation follows from the findings."
    )
    evidence: Evidence


class BusinessMetric(BaseModel):
    name: str = Field(description="e.g. 'Total Addressable Market', 'Projected Revenue Growth'")
    value: str = Field(description="e.g. '$2.3B', '15% YoY'")
    evidence: Evidence


class RiskOrConcern(BaseModel):
    description: str
    severity: Optional[str] = Field(
        default=None, description="agent's assessment: low / medium / high"
    )
    evidence: Evidence


class ProcessingMetadata(BaseModel):
    pages_or_sections_processed: int
    extraction_warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during parsing or analysis, "
        "e.g. 'Table on page 6 could not be parsed and was skipped', "
        "'Page 9 appears to be a scanned image with no extractable text'.",
    )


class StrategyDocumentAnalysis(BaseModel):
    document_title: str
    document_type: str = Field(
        description="e.g. 'Corporate Strategy Document', 'Market-Entry Study', "
        "'Business Plan', 'Transformation Roadmap', 'Market Research Report', "
        "or 'Other' with a brief description."
    )
    executive_summary: str
    strategic_objective: str
    key_findings: List[Finding]
    main_recommendations: List[Recommendation]
    business_metrics: List[BusinessMetric]
    assumptions: List[str] = Field(
        description="Assumptions the document itself relies on (stated or implied), "
        "not assumptions made by the agent."
    )
    risks_or_concerns: List[RiskOrConcern]
    missing_or_unclear_information: List[str] = Field(
        description="Information a consulting reviewer would expect but that is "
        "absent, ambiguous, or contradictory in the document."
    )
    recommended_follow_up_questions: List[str]
    overall_confidence: ConfidenceLevel = Field(
        description="Agent's holistic confidence in the completeness/reliability "
        "of this analysis, independent of per-claim confidence."
    )
    processing_metadata: ProcessingMetadata


# JSON schema handed to the model in the prompt (kept separate from the
# Pydantic model's own .model_json_schema() so we can trim/annotate it for
# prompting purposes if needed later).
def get_schema_for_prompt() -> dict:
    return StrategyDocumentAnalysis.model_json_schema()
