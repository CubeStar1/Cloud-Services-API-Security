
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

import glob
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.model_selection import train_test_split

from .csv_utils import convert_all_raw_json_to_csv
from .path_config import PATHS

load_dotenv()

from pydantic import BaseModel, Field

CONFIG: Dict[str, Any] = {
    "test_split": 0.2,
    "groq_model": "moonshotai/kimi-k2-instruct-0905",
}

class TrafficClassification(BaseModel):
    service: str = Field(description="The API service being accessed (e.g., 'Google Drive', 'Slack', 'Unknown')")
    activity: str = Field(description="The activity performed (e.g., 'Upload', 'Download', 'Login', 'Unknown')")


def print_status(msg: str) -> None:  
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode("ascii"), flush=True)


def metadata_path() -> Path:
    return Path(PATHS["metadata_file"])


def load_metadata() -> Dict[str, Any]:
    if metadata_path().exists():
        return json.loads(metadata_path().read_text())
    return {"processed_files": {}, "labelling_progress": {}}


def save_metadata(meta: Dict[str, Any]) -> None:
    metadata_path().write_text(json.dumps(meta, indent=4))
    print_status("[+] Metadata updated")


def load_combined_dataset() -> pd.DataFrame:
    """Concatenate all CSVs in csv_folder into one DataFrame (generate if needed)."""
    csv_dir = Path(PATHS["csv_folder"])
    if not list(csv_dir.glob("*.csv")):
        convert_all_raw_json_to_csv()
    frames = [pd.read_csv(p) for p in csv_dir.glob("*.csv")]
    if not frames:
        raise FileNotFoundError("No CSV files found after conversion.")
    return pd.concat(frames, ignore_index=True)


def split_and_save(df: pd.DataFrame) -> Tuple[Path, Path]:
    train_df, test_df = train_test_split(
        df, test_size=CONFIG["test_split"], random_state=42
    )
    labelled_dir = Path(PATHS["labelled_folder"])
    labelled_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(PATHS["train_file"])
    test_path = Path(PATHS["test_file"])
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def get_groq_classification(client: Groq, prompt: str) -> Tuple[str, str]:
    for _ in range(3):  # simple retry
        try:
            completion = client.chat.completions.create(
                model=CONFIG["groq_model"],
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an API traffic classification expert. Classify the following HTTP request into a specific Service and Activity."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "traffic_classification",
                        "schema": TrafficClassification.model_json_schema()
                    }
                }
            )
            content = completion.choices[0].message.content.strip()
            
            # Use Pydantic to validate
            classification = TrafficClassification.model_validate(json.loads(content))
            return classification.service, classification.activity

        except Exception as err:  
            print_status(f"[!] Groq error: {err}. Retrying...")
            time.sleep(2)
    return "Unknown", "Unknown"


def create_prompt(
    row: pd.Series, 
    services: List[str] | None = None, 
    activities: List[str] | None = None
) -> str:
    activity_str = ", ".join(activities) if activities else "Unknown"
    service_str = ", ".join(services) if services else "Unknown"
    
    classification_guide = ""
    if services and activities:
        classification_guide = (
            f"Possible Services: [{service_str}]\n"
            f"Possible Activities: [{activity_str}]\n"
        )
    
    return (
        "Classify the service and activity for this request:\n"
        f"Host: {row.get('headers_Host', 'N/A')}\n"
        f"Method: {row.get('method', 'N/A')}\n"
        f"URL: {row.get('url', 'N/A')}\n\n"
        f"{classification_guide}"
    )


def label_file(
    csv_path: Path, 
    client: Groq, 
    meta: Dict[str, Any],
    services: List[str] | None = None,
    activities: List[str] | None = None
) -> pd.DataFrame:
    name = csv_path.name
    print_status(f"[*] Processing {name}")

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    progress = meta.setdefault("labelling_progress", {}).setdefault(name, 0)

    for idx in range(progress, total_rows):
        row = df.iloc[idx]
        if pd.isna(row.get("predicted_service")) or pd.isna(row.get("predicted_activity")):
            prompt = create_prompt(row, services, activities)
            service, activity = get_groq_classification(client, prompt)
            df.at[idx, "predicted_service"] = service
            df.at[idx, "predicted_activity"] = activity

        if (idx + 1) % 20 == 0 or idx == total_rows - 1:
            meta["labelling_progress"][name] = idx + 1
            save_metadata(meta)
            df.to_csv(csv_path, index=False)
            print_status(f"    ↳ saved progress ({idx + 1}/{total_rows})")

    print_status(f"[+] Completed {name}")
    return df



def run_full_labelling(
    api_key: str | None = None,
    services: List[str] | None = None,
    activities: List[str] | None = None
) -> Dict[str, Any]:
    """Convert raw logs ➜ split ➜ label with Groq and return summary stats."""

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set and no api_key provided")

    convert_all_raw_json_to_csv()
    combined_df = load_combined_dataset()
    train_path, test_path = split_and_save(combined_df)

    meta = load_metadata()
    client = get_groq_client(api_key)

    train_df = label_file(train_path, client, meta, services, activities)
    test_df = label_file(test_path, client, meta, services, activities)

    save_metadata(meta)

    summary = {
        "train_service_counts": train_df["predicted_service"].value_counts().to_dict(),
        "train_activity_counts": train_df["predicted_activity"].value_counts().to_dict(),
        "test_service_counts": test_df["predicted_service"].value_counts().to_dict(),
        "test_activity_counts": test_df["predicted_activity"].value_counts().to_dict(),
    }

    print_status("[+] Labelling pipeline completed")
    return {"success": True, "message": summary}

