# pip install google-genai pydantic rapidfuzz
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
import os
from rapidfuzz import process, fuzz

class IssueExtraction(BaseModel):
    theme: str = Field(
        description="Short free-text label for the high-level theme (e.g., 'timeliness', 'duplicate trades')."
    )
    theme_normalized: Optional[str] = Field(
        default=None,
        description="Optional: normalized theme mapped to your preferred taxonomy (leave null if not confidently mappable)."
    )
    theme_confidence: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Optional: model's confidence (0-1) in the normalized theme."
    )
    issue_summary: str = Field(
        description="One-sentence summary of the issue in plain English."
    )
    data_type: Optional[str] = Field(
        default=None,
        description="Business data category (e.g., Trades, Orders, Positions, Prices, Reference Data, Corporate Actions, Risk, PnL)."
    )
    source_system: Optional[str] = Field(
        default=None,
        description="Primary upstream/source system if present (e.g., Murex, Calypso, Aladdin, Snowflake, Kafka topic)."
    )
    data_attribute: Optional[List[str]] = Field(
        default=None,
        description="Specific fields affected (e.g., trade_id, execution_time, price, quantity, currency, isin)."
    )

# ---------- Gemini client ----------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTIONS = (
    "Extract fields from capital markets DCRM issue descriptions. "
    "Be concise and factual. If a field is unknown, return null. "
    "Set 'theme' to a short free-text label. Do not invent systems/attributes."
)

FEW_SHOTS = [
    {
        "issue": (
            "DCRM: T+1 PnL breaks because positions from Murex arrive after 7am EST. "
            "Missing close_price for multiple ISINs in the reference feed."
        ),
        "label": {
            "theme": "timeliness / missing values",
            "theme_normalized": None,
            "theme_confidence": None,
            "issue_summary": "PnL breaks due to late Murex positions and missing close_price values.",
            "data_type": "Positions/PnL",
            "source_system": "Murex",
            "data_attribute": ["close_price", "isin"]
        }
    },
    {
        "issue": (
            "Client flagged duplicate trades in Snowflake fact_trades; trade_id repeats "
            "after intraday reload (Kafka replay suspected)."
        ),
        "label": {
            "theme": "duplication",
            "theme_normalized": None,
            "theme_confidence": None,
            "issue_summary": "Duplicate trades caused by intraday reload/replay into Snowflake.",
            "data_type": "Trades",
            "source_system": "Snowflake",
            "data_attribute": ["trade_id"]
        }
    },
]

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

def extract_issue_fields(text: str) -> IssueExtraction:
    prompt = _build_prompt(text)
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": IssueExtraction,  # strict structured output, no enum
        },
    )
    return resp.parsed  # -> IssueExtraction
