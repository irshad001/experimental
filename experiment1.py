# pip install google-cloud-aiplatform pydantic

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ---------- 0) Vertex AI init ----------
# Set your GCP project & region (env or inline)
PROJECT_ID = os.getenv("GCP_PROJECT", "your-gcp-project-id")
LOCATION   = os.getenv("GCP_LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

# ---------- 1) Define your output schema ----------
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
    data_type: Optional[str] = Field(
        default=None,
        description="Business data category, e.g., Trades, Orders, Positions, Prices, Reference Data, Corporate Actions, Client, Instrument, Risk, PnL.",
    )
    source_system: Optional[str] = Field(
        default=None,
        description="Primary upstream/source system named or implied in the description (e.g., Murex, Calypso, Aladdin, Markit EDM, Snowflake, Kafka topic).",
    )
    data_attribute: Optional[List[str]] = Field(
        default=None,
        description="Specific attributes/columns/fields affected (e.g., trade_id, execution_time, price, quantity, currency, isin, counterparty).",
    )

# ---------- 2) Model ----------
# Pick an available Vertex model name in your region.
# Examples: "gemini-2.5-flash", "gemini-1.5-flash-001", etc.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = GenerativeModel(MODEL_NAME)

# ---------- 3) Few-shot guidance (optional but recommended) ----------
FEW_SHOTS = [
    {
        "issue": (
            "DCRM: Traders report T+1 breaks in PnL because positions from Murex "
            "arrive after 7am EST. Missing 'close_price' for several ISINs. "
            "Markit reference feed shows gaps."
        ),
        "label": IssueExtraction(
            theme="Timeliness/Latency",
            issue_summary="PnL breaks due to late Murex positions and missing close_price values.",
            data_type="Positions/PnL",
            source_system="Murex",
            data_attribute=["close_price", "isin"],
        ).model_dump(),
    },
    {
        "issue": (
            "Client flagged duplicate trades in Snowflake fact_trades; trade_id repeats "
            "after intraday reload. Root cause unknown; Kafka topic may replay."
        ),
        "label": IssueExtraction(
            theme="Duplication",
            issue_summary="Duplicate trades caused by intraday reload/replay into Snowflake.",
            data_type="Trades",
            source_system="Snowflake",
            data_attribute=["trade_id"],
        ).model_dump(),
    },
    {
        "issue": (
            "Corporate actions mapping missing for multiple tickers; instrument reference "
            "record lacks FIGI; downstream risk calc fails."
        ),
        "label": IssueExtraction(
            theme="Reference Data/Mapping",
            issue_summary="Missing corporate action mappings and FIGI cause downstream risk calc failures.",
            data_type="Reference Data / Corporate Actions",
            source_system=None,
            data_attribute=["figi", "ticker"],
        ).model_dump(),
    },
]

SYSTEM_INSTRUCTIONS = (
    "You extract fields from capital markets DCRM issue descriptions. "
    "Be concise, factual, and prefer values explicitly present in text. "
    "If unknown, leave field null (do not invent)."
)

def _build_prompt(text: str) -> str:
    shots = []
    for s in FEW_SHOTS:
        shots.append(f"Issue:\n{s['issue']}\nJSON:\n{s['label']}\n")
    return (
        SYSTEM_INSTRUCTIONS
        + "\n\nExamples:\n"
        + "\n---\n".join(shots)
        + "\n\nNow extract from this Issue:\n"
        + text
    )

# ---------- 4) Single-text extractor ----------
def extract_issue_fields(text: str) -> IssueExtraction:
    prompt = _build_prompt(text)

    # In Vertex AI, structured output is provided via GenerationConfig.
    # The SDK supports Pydantic BaseModel as response_schema.
    resp = model.generate_content(
        contents=prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=IssueExtraction,  # <- strict structured output
            temperature=0.0,                  # determinism
        ),
    )

    # Vertex returns a text block containing JSON that conforms to the schema.
    # Newer SDK versions may give you a parsed object via helpers, but this
    # pattern (construct via Pydantic) is robust:
    return IssueExtraction.model_validate_json(resp.text)

# ---------- 5) Batch helper ----------
def extract_batch(issues: List[str]) -> List[IssueExtraction]:
    out: List[IssueExtraction] = []
    for t in issues:
        try:
            out.append(extract_issue_fields(t))
        except Exception:
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

# ---------- 6) Quick test ----------
if __name__ == "__main__":
    sample = (
        "Operations observed missing execution_time and currency on ~3% of FX trades "
        "originating from Calypso overnight load; downstream reconciliation flagged breaks."
    )
    result = extract_issue_fields(sample)
    print(result.model_dump())
