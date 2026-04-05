#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from components.inference.manifest import (
    DEFAULT_CPU_REQUEST,
    DEFAULT_IMAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_MEMORY_REQUEST,
    DEFAULT_LOCAL_MODEL_MOUNT_PATH,
    DEFAULT_MODEL_IDS,
    DEFAULT_NAMESPACE,
    DEFAULT_REPLICAS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TORCH_THREADS,
    dump_qwen_inference_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Qwen inference Deployment/Service manifest.")
    parser.add_argument("--role", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--model-id", help="Hugging Face model ID or other model identifier.")
    parser.add_argument("--model-uri", help="Optional logical model URI to annotate on the Deployment.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--cpu-request", default=DEFAULT_CPU_REQUEST)
    parser.add_argument("--memory-request", default=DEFAULT_MEMORY_REQUEST)
    parser.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT)
    parser.add_argument("--local-model-source-path", help="Absolute host/node path to a local model directory.")
    parser.add_argument(
        "--local-model-mount-path",
        default=DEFAULT_LOCAL_MODEL_MOUNT_PATH,
        help="Container mount path to use when loading a local model directory.",
    )
    args = parser.parse_args()

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.role]
    sys.stdout.write(
        dump_qwen_inference_manifests(
            role=args.role,
            model_id=model_id,
            model_uri=args.model_uri,
            namespace=args.namespace,
            image=args.image,
            replicas=args.replicas,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            torch_threads=args.torch_threads,
            cpu_request=args.cpu_request,
            memory_request=args.memory_request,
            memory_limit=args.memory_limit,
            local_model_source_path=args.local_model_source_path,
            local_model_mount_path=args.local_model_mount_path,
        )
    )


if __name__ == "__main__":
    main()
