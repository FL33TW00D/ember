import argparse
from coremltools import ComputeUnit
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision
from .serve import serve
from .create import create 

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


def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "create", help="Create CoreML model for embeddings from a SentenceTransformer model."
    )

    generate_parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Provide location to save exported models.",
    )
    generate_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference. Default is 64.",
    )
    generate_parser.add_argument(
        "--min-deployment-target",
        type=lambda x: getattr(AvailableTarget, x),
        choices=[target.name for target in AvailableTarget],
        default=AvailableTarget.iOS17,
        help="Minimum deployment target for CoreML model.",
    )
    generate_parser.add_argument(
        "--compute-units",
        type=lambda x: getattr(ComputeUnit, x),
        default=ComputeUnit.ALL,
        help="Which compute units to target for CoreML model.",
    )
    generate_parser.add_argument(
        "--precision",
        type=lambda x: getattr(ComputePrecision, x),
        choices=[p for p in ComputePrecision],
        default=ComputePrecision.FLOAT16,
        help="Precision used for computation in CoreML model, FLOAT16 can cause precision issues vs PyTorch model.",
    )
    generate_parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="Tolerance for comparing PyTorch and CoreML model outputs.",
    )
    generate_parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=35,
        help="PSNR threshold for comparing PyTorch and CoreML model outputs.",
    )
    generate_parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the exported model to Hugging Face Hub",
        default=False,
    )
    generate_parser.add_argument(
        "--hf-repo-id",
        type=str,
        help="Hugging Face Hub repository ID (e.g., 'username/repo-name')",
        required=False,
    )
    generate_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model outputs between PyTorch and CoreML",
        default=True,
    )
    generate_parser.add_argument(
        "--force-ane",
        action="store_true",
        help="Fail if model is not entirely ANE resident",
        default=False,
    )

    serve_parser = subparsers.add_parser(
        "serve", help="Serve the CoreML model locally."
    )

    return parser


def main():
    parser = argparse.ArgumentParser(
        description="Script to export coreml package file for sentence-transformers models"
    )
    parser = parse_args(parser)
    args = parser.parse_args()

    match args.command:
        case "create":
            create(
                args.output_dir,
                args.batch_size,
                args.min_deployment_target,
                args.compute_units,
                args.precision,
                args.tolerance,
                args.psnr_threshold,
                args.push_to_hub,
                args.hf_repo_id,
                args.validate,
                args.force_ane,
            )
        case "serve":
            serve()
