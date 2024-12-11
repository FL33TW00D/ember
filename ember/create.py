import warnings

import numpy as np
import torch

import questionary

from huggingface_hub import HfApi, create_repo, repo_exists

import coremltools as ct
from coremltools import ComputeUnit
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision
from coremlprofiler import CoreMLProfiler, ComputeDevice

from sentence_transformers import SentenceTransformer

import json
import os

warnings.filterwarnings("ignore")

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model.forward(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return outputs["sentence_embedding"]


class ModelChoice:
    def __init__(self, org, name, description):
        self.org = org
        self.name = name
        self.description = description

    def __str__(self):
        return f"{self.org}/{self.name} - {self.description}"

    def __repr__(self):
        return f"{self.org}/{self.name}"

    def as_pkg(self):
        return f"{self.name}.mlpackage"


MODEL_CHOICES = [
    ModelChoice(
        "sentence-transformers",
        "all-mpnet-base-v2",
        "100M parameter model. Intented to be used as a sentence and short paragraph encoder",
    ),
    ModelChoice(
        "sentence-transformers",
        "all-MiniLM-L12-v2",
        "50M parameter model. Intended to be used as a sentence and short paragraph encoder",
    ),
    ModelChoice(
        "sentence-transformers",
        "all-MiniLM-L6-v2",
        "22M parameter model. Intended to be used as a sentence and short paragraph encoder",
    ),
    ModelChoice(
        "BAAI",
        "bge-small-en-v1.5",
        "~33M parameter model.",
    ),
    ModelChoice(
        "BAAI",
        "bge-base-en-v1.5",
        "~110M parameter model.",
    ),
    ModelChoice(
        "BAAI",
        "bge-large-en-v1.5",
        "~330M parameter model.",
    ),
    ModelChoice(
        "mixedbread-ai",
        "mxbai-embed-large-v1",
        "~330M parameter model.",
    ),
    ModelChoice(
        "nomic-ai",
        "nomic-embed-text-v1.5",
        "~137M parameter model.",
    ),
    ModelChoice(
        "intfloat",
        "multilingual-e5-small",
        "~118M parameter model.",
    ),
    ModelChoice(
        "intfloat",
        "multilingual-e5-base",
        "~275M parameter model.",
    ),
    ModelChoice(
        "intfloat",
        "multilingual-e5-large",
        "~560M parameter model.",
    ),
    ModelChoice(
        "Snowflake",
        "snowflake-arctic-embed-m-v2.0",
        "~305M parameter model.",
    ),
    ModelChoice(
        "Snowflake",
        "snowflake-arctic-embed-l-v2.0",
        "~570M parameter model.",
    ),
]


def select_model() -> ModelChoice:
    choices = [str(choice) for choice in MODEL_CHOICES]
    choices.append("Custom Model")

    selected = questionary.select(
        "Choose a model to export:",
        choices=choices,
        style=questionary.Style(
            [
                ("qmark", "fg:green"),
                ("question", "bold"),
                ("choice", "fg:cyan"),
                ("pointer", "fg:green"),
                ("highlighted", "fg:green"),
            ]
        ),
    ).ask()
    if selected.startswith("Custom"):
        custom_id = questionary.text("Enter the model ID (org/name, use the ⎘ button on the Hub 🤗)").ask()
        if "/" not in custom_id:
            raise ValueError("Invalid model ID. Please use the format org/name")

        custom_org, custom_model = custom_id.split("/")
        return ModelChoice(custom_org, custom_model, "Custom Model")
    else:
        return next(choice for choice in MODEL_CHOICES if str(choice) == selected)


# Adapted from: https://github.com/apple/ml-stable-diffusion/blob/cf16df8207dfcba685a9391bad04f7402ea87b73/python_coreml_stable_diffusion/torch2coreml.py#L59
def compute_psnr(a, b):
    """Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects"""
    max_b = np.abs(b).max()
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10((max_b + 1e-5) / (np.sqrt(mse) + 1e-10))


def validate_model(
    pytorch_model, coreml_model, input_ids, attention_mask, tolerance, psnr_threshold
):
    with torch.no_grad():
        pytorch_output = pytorch_model(input_ids, attention_mask).numpy()

    coreml_output = coreml_model.predict(
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        }
    )["sentence_embeddings"]

    max_diff = np.max(np.abs(pytorch_output - coreml_output))
    avg_diff = np.mean(np.abs(pytorch_output - coreml_output))
    psnr = compute_psnr(pytorch_output, coreml_output)

    if (
        np.allclose(pytorch_output, coreml_output, atol=tolerance)
        and psnr > psnr_threshold
    ):
        print(
            f"{bcolors.OKGREEN}"
            f"Model validation successful. Pytorch and CoreML match. Max diff: {max_diff}, Avg diff: {avg_diff}, PSNR: {psnr} > {psnr_threshold}"
            f"{bcolors.ENDC}"
        )

    else:
        raise ValueError(
            f"{bcolors.FAIL}"
            f"Model validation failed. PyTorch and CoreML outputs do not match within {tolerance}, or PSNR < {psnr_threshold}"
            f"Max diff: {max_diff}, Avg diff: {avg_diff}, PSNR: {psnr}"
            f"{bcolors.ENDC}"
        )


def push_to_hub(model_path: str, repo_id: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set")

    api = HfApi()

    print("Pushing model to Hugging Face Hub...")
    if not repo_exists(repo_id, token=token):
        create_repo(repo_id, private=True, token=token)

    model_path = os.path.abspath(model_path)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        token=token,
    )
    print(f"Model pushed to Hugging Face Hub: {repo_id}")


def coreml_precision_to_torch(precision: ComputePrecision) -> torch.dtype:
    if precision == ComputePrecision.FLOAT32:
        return torch.float32
    elif precision == ComputePrecision.FLOAT16:
        return torch.float16
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def create(
    output_dir: str,
    batch_size: int,
    min_deployment_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
    tolerance: float,
    psnr_threshold: float,
    should_push: bool,
    hf_repo_id: str,
    validate: bool,
    force_ane: bool,
):
    selected = select_model()

    if os.path.exists(f"{selected.as_pkg()}"):
        print(f"Model {selected.as_pkg()} already exists. Skipping.")
        return

    if should_push and hf_repo_id is None:
        raise ValueError("--hf-repo-id is required to push to Hugging Face Hub")

    os.makedirs(output_dir, exist_ok=True)

    device_str = "cpu"
    device = torch.device(device_str)

    model = SentenceTransformer(
        repr(selected), device=device_str, trust_remote_code=True
    )
    max_length = model.get_max_seq_length()
    print(f"Model max sequence length: {max_length}")

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    input_shape = (batch_size, max_length)

    vocab_dt = torch.int32
    input_ids = torch.randint(
        0, model.tokenizer.vocab_size, input_shape, device=device, dtype=vocab_dt
    )
    attention_mask = torch.ones(input_shape, device=device, dtype=vocab_dt)

    traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=input_shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="sentence_embeddings", dtype=np.float32),
        ],
        minimum_deployment_target=min_deployment_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    if precision == ComputePrecision.FLOAT16:
        wrapped_model = wrapped_model.half()

    if validate:
        validate_model(
            wrapped_model, mlmodel, input_ids, attention_mask, tolerance, psnr_threshold
        )

    mlmodel.save(selected.as_pkg())

    profiler = CoreMLProfiler(selected.as_pkg())

    summary = profiler.device_usage_summary()
    if force_ane:
        if summary[ComputeDevice.CPU] > 0 or summary[ComputeDevice.GPU] > 0:
            raise ValueError(
                f"Model contains operations that cannot run on ANE."
                f"{summary}"
                "You can allow this behaviour running with --force-ane"
            )

    print(f"\n{profiler.device_usage_summary_chart()}\n")

    with open(f"{selected.as_pkg()}/ember_config.json", "w") as json_file:
        json.dump({"batch_size": batch_size, "max_length": max_length}, json_file)

    if should_push:
        push_to_hub(selected.as_pkg(), hf_repo_id)
