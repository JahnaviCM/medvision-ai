import json
import zipfile
from pathlib import Path

model_path = Path("models/pneumonia_cnn_model.keras")
fixed_path = Path("models/pneumonia_cnn_model_fixed.keras")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

def clean_config(obj):
    if isinstance(obj, dict):
        # Remove only null quantization settings that older
        # Keras layers may reject.
        obj.pop("quantization_config", None)
        for value in obj.values():
            clean_config(value)
    elif isinstance(obj, list):
        for value in obj:
            clean_config(value)

with zipfile.ZipFile(model_path, "r") as source:
    files = {
        name: source.read(name)
        for name in source.namelist()
    }

config_name = next(
    (name for name in files if name.endswith("config.json")),
    None
)

if config_name is None:
    raise ValueError("Could not find config.json inside the model.")

config = json.loads(files[config_name].decode("utf-8"))
clean_config(config)
files[config_name] = json.dumps(config).encode("utf-8")

with zipfile.ZipFile(fixed_path, "w") as output:
    for name, data in files.items():
        output.writestr(name, data)

print("Fixed model archive created:", fixed_path)
print("Original model has been preserved.")