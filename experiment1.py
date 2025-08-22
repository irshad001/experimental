# pip install -U google-cloud-aiplatform pydantic

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ---------- 0) Vertex init ----------
# Set your GCP project and region (must support the selected Gemini model)
PROJECT_ID = os.getenv("GCP_PROJECT", "your-gcp-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ---------- 1) Output schema ----------
ThemeEnum = Literal[
    "Data Quality",
    "Lineage/Provenance",
    "Timeliness/Latency",
    "Completeness/Missing Data",
    "Conformity/Standards",
    "Duplication",
    "Access/Permissioning",
    "Reference Data/Mapping",
    "Calculation/Derivation",
    "Controls/Breaks/Reconciliation",
    "Other/Unclear",
]

class IssueExtraction(BaseModel):
    theme: ThemeEnum = Field(description="High-level theme of the issue.")
    issue_summary: str = Field(description="One-sentence summary of the issue in plain English.")
    data_type: Optional[str] = None
    source_system: Optional[str] = None
    data_attribute: Optional[List[str]] = None

# Explicit JSON schema to satisfy vertexai 1.71.1
ISSUE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "theme": {
            "type": "string",
            "enum": [
                "Data Quality",
                "Lineage/Provenance",
                "Timeliness/Latency",
                "Completeness/Missing Data",
                "Conformity/Standards",
                "Duplication",
                "Access/Permissioning",
                "Reference Data/Mapping",
                "Calculation/Derivation",
                "Controls/Breaks/Reconciliation",
                "Other/Unclear",
            ],
        },
        "issue_summary": {"type": "string"},
        "data_type": {"type": "string", "nullable": True},
        "source_system": {"type": "string", "nullable": True},
        "data_attribute": {
            "type": "array",
            "items": {"type": "string"},
            "nullable": True,
        },
    },
    "required": ["theme", "issue_summary"],
    "additionalProperties": False,
}

# ---------- 2) Model ----------
# Use a Gemini model available in your region
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = GenerativeModel(MODEL_NAME)

# ---------- 3) Few-shot examples ----------
FEW_SHOTS = [
    {
        "issue": (
            "DCRM: Traders report T+1 breaks in PnL because positions from Murex "
            "arrive after 7am EST. Missing 'close_price' for several ISINs. "
            "Markit reference feed shows gaps."
        ),
        "label": {
            "theme": "Timeliness/Latency",
            "issue_summary": "PnL breaks due to late Murex positions and missing close_price values.",
            "data_type": "Positions/PnL",
            "source_system": "Murex",
            "data_attribute": ["close_price", "isin"],
        },
    },
    {
        "issue": (
            "Client flagged duplicate trades in Snowflake fact_trades; trade_id repeats "
            "after intraday reload. Kafka replay suspected."
        ),
        "label": {
            "theme": "Duplication",
            "issue_summary": "Duplicate trades due to intraday reload/replay in Snowflake.",
            "data_type": "Trades",
            "source_system": "Snowflake",
            "data_attribute": ["trade_id"],
        },
    },
    {
        "issue": (
            "Corporate actions mapping missing for multiple tickers; instrument reference "
            "record lacks FIGI; downstream risk calc fails."
        ),
        "label": {
            "theme": "Reference Data/Mapping",
            "issue_summary": "Missing corporate action mappings and FIGI cause downstream risk calc failures.",
            "data_type": "Reference Data / Corporate Actions",
            "source_system": None,
            "data_attribute": ["figi", "ticker"],
        },
    },
    {
        "issue": (
            "Intraday prices show currency mismatch; EUR trades stored with USD currency "
            "in the market data snapshot leading to valuation errors."
        ),
        "label": {
            "theme": "Conformity/Standards",
            "issue_summary": "Currency code mismatch in intraday prices leads to valuation errors.",
            "data_type": "Prices/Market Data",
            "source_system": None,
            "data_attribute": ["currency", "price"],
        },
    },
]

SYSTEM_INSTRUCTIONS = (
    "You extract fields from capital markets DCRM issue descriptions. "
    "Be concise, factual, and prefer values explicitly present in text. "
    "If a field is unknown, return null (do not invent)."
)

# ---------- 4) Prompt builder ----------
def _build_prompt(text: str) -> str:
    examples = []
    for s in FEW_SHOTS:
        examples.append(
            "Issue:\n"
            + s["issue"]
            + "\nJSON:\n"
            + json.dumps(s["label"], ensure_ascii=False)
        )
    return (
        SYSTEM_INSTRUCTIONS
        + "\n\nExamples:\n"
        + "\n---\n".join(examples)
        + "\n\nNow extract from this Issue:\n"
        + text
    )

# ---------- 5) Single-text extractor ----------
def extract_issue_fields(text: str) -> IssueExtraction:
    prompt = _build_prompt(text)
    resp = model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=ISSUE_SCHEMA,
            temperature=0.0,  # deterministic output
        ),
    )
    # Validate and parse the returned JSON with Pydantic
    return IssueExtraction.model_validate_json(resp.text)

# ---------- 6) Batch helper ----------
def extract_batch(issues: List[str]) -> List[IssueExtraction]:
    out: List[IssueExtraction] = []
    for t in issues:
        try:
            out.append(extract_issue_fields(t))
        except Exception:
            # Return a minimal record with theme Other/Unclear on failure
            out.append(
                IssueExtraction(
                    theme="Other/Unclear",
                    issue_summary=str(t)[:200],
                    data_type=None,
                    source_system=None,
                    data_attribute=None,
                )
            )
    return out

# ---------- 7) Quick test ----------
if __name__ == "__main__":
    sample = (
        "Operations observed missing execution_time and currency on ~3% of FX trades "
        "originating from Calypso overnight load; downstream reconciliation flagged breaks."
    )
    # Uncomment the two lines below if you have Vertex AI configured and valid permissions
    # result = extract_issue_fields(sample)
    # print(result.model_dump())
