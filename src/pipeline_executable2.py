# pipeline_executable.py
#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import numba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─── Logging setup ───────────────────────────────────────────────────────
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

# ─── Arg parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run the BIGFISH/ICC pipeline with optional GPU/CPU selection"
)
parser.add_argument(
    "pipeline_location",
    help="Path to the pipeline JSON config file"
)
parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default=os.environ.get("DEVICE", "cpu"),
    help="‘cuda’ or ‘cpu’. Falls back to $DEVICE or cpu"
)
args = parser.parse_args()
pipeline_location = os.path.normpath(args.pipeline_location)
device = args.device
torch_device = torch.device(device)

# ─── Report device ───────────────────────────────────────────────────────
print(f"→ Using device: {device}")
print(f"CUDA available? {torch.cuda.is_available()}")
if device == "cuda" and torch.cuda.is_available():
    ngpu = torch.cuda.device_count()
    print(f"→ GPU count: {ngpu}")
    for i in range(ngpu):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("→ Running on CPU")

# ─── Load and configure pipeline ─────────────────────────────────────────
with open(pipeline_location, "r") as f:
    pipeline_dict = json.load(f)

# Preserve old behavior for display and NAS connection:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
connection_config = os.path.join(repo_path, "config_nas.yml")

pipeline_dict["params"]["display_plots"] = False
pipeline_dict["params"]["connection_config_location"] = connection_config

# Convert illumination profiles if present
if "illumination_profiles" in pipeline_dict["params"]:
    pipeline_dict["params"]["illumination_profiles"] = np.array(
        pipeline_dict["params"]["illumination_profiles"]
    )

# Inject our device choice
pipeline_dict["params"]["device"] = device

steps = pipeline_dict["steps"]
experiment_locations = pipeline_dict["params"]["initial_data_location"]

print("Steps:", steps)
print("Parameters:", pipeline_dict["params"])

# Make sure our code can load the src/ folder
src_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.Parameters import Parameters
from src import StepClass
from src.Pipeline import Pipeline

# ─── Run ──────────────────────────────────────────────────────────────────
pipeline = Pipeline(
    experiment_location=experiment_locations,
    parameters=pipeline_dict["params"],
    steps=steps
)
pipeline.run()