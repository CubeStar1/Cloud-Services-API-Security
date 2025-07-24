from pathlib import Path

# Project root is three levels up from this file (backend/utils/path_config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reuse existing data directory in project root
DATA_DIR = PROJECT_ROOT / "backend" / "data"

PATHS = {
    "data_folder": DATA_DIR,
    "raw_json_folder": DATA_DIR / "logs" / "raw-json",
    "csv_folder": DATA_DIR / "logs" / "csv",
    "labelled_folder": DATA_DIR / "labelled",
    "metadata_file": DATA_DIR / "labelled" / "metadata.json",

    # RFC Python training specific paths
    "rfc_python_train_input": DATA_DIR / "output" / "codebert" / "predictions",
    "rfc_python_train_models": DATA_DIR / "output" / "rfc" / "models",
    "rfc_python_train_test": DATA_DIR / "output" / "rfc" / "test",

    # RFC C code generation specific paths
    "rfc_codegen_input_folder": DATA_DIR / "output" / "codebert" / "predictions",
    "rfc_codegen_output_folder": DATA_DIR / "output" / "rfc" / "codegen",

    # RFC Inference specific paths
    "rfc_python_inference_input_folder": DATA_DIR / "output" / "codebert" / "predictions",
    "rfc_python_inference_output_folder": DATA_DIR / "output" / "rfc" / "inference",
    "rfc_python_inference_models": DATA_DIR / "output" / "rfc" / "models",
    
}

# Ensure directories exist when module is imported
for key, path in PATHS.items():
    if "_file" not in key:  # skip files
        Path(path).mkdir(parents=True, exist_ok=True)
