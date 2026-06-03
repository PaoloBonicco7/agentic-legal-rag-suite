#!/usr/bin/env python3
"""List available models on Utopia Ollama server."""

import os
import json
import sys
from pathlib import Path
from typing import Optional

import requests


def load_env_file(path: str = ".env") -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    env_path = Path(path)

    if not env_path.exists():
        print(f"Warning: {path} not found")
        return env_vars

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    return env_vars


def list_ollama_models(
    base_url: str,
    api_key: str,
    verbose: bool = False
) -> Optional[list[dict]]:
    """
    Fetch list of available models from Ollama server.

    Args:
        base_url: Base URL of Ollama server (e.g., https://utopia.hpc4ai.unito.it/api)
        api_key: Bearer token for authentication
        verbose: Print detailed information

    Returns:
        List of model dictionaries or None if request failed
    """
    # Construct the tags endpoint
    # Remove trailing /api if present and construct full Ollama endpoint
    base = base_url.rstrip("/api").rstrip("/")
    url = f"{base}/ollama/api/tags"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        if verbose:
            print(f"Fetching models from: {url}")

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = data.get("models", [])

        if verbose:
            print(f"Found {len(models)} model(s)\n")

        return models

    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}", file=sys.stderr)
        return None


def print_models_table(models: list[dict], currently_used: dict[str, str]) -> None:
    """Print models in a formatted table."""
    if not models:
        print("No models found")
        return

    print(f"{'Model Name':<40} {'Size':<15} {'Used In':<20}")
    print("-" * 75)

    for model in models:
        name = model.get("name", "Unknown")

        # Get size in GB if available
        size_bytes = model.get("size", 0)
        size_gb = size_bytes / (1024**3) if size_bytes else 0
        size_str = f"{size_gb:.2f} GB" if size_gb else "Unknown"

        # Check if this model is currently used
        used_in = []
        for config_name, config_model in currently_used.items():
            if config_model in name:
                used_in.append(config_name)

        used_str = ", ".join(used_in) if used_in else "-"

        print(f"{name:<40} {size_str:<15} {used_str:<20}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="List available models on Utopia Ollama server"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--base-url",
        help="Override base URL from .env",
    )
    parser.add_argument(
        "--api-key",
        help="Override API key from .env",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load environment
    env_vars = load_env_file(args.env_file)
    os.environ.update(env_vars)

    # Get configuration
    base_url = args.base_url or env_vars.get("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api")
    api_key = args.api_key or env_vars.get("UTOPIA_API_KEY")

    if not api_key:
        print("Error: UTOPIA_API_KEY not found in environment or .env file", file=sys.stderr)
        sys.exit(1)

    # Fetch models
    models = list_ollama_models(base_url, api_key, verbose=args.verbose)

    if models is None:
        sys.exit(1)

    # Prepare currently used models info
    currently_used = {
        "CHAT": env_vars.get("UTOPIA_CHAT_MODEL", ""),
        "EMBED": env_vars.get("UTOPIA_EMBED_MODEL", ""),
    }

    # Output
    if args.json:
        print(json.dumps(models, indent=2))
    else:
        print("🔍 Available Models on Utopia Ollama Server")
        print(f"Server: {base_url}")
        print()
        print_models_table(models, currently_used)
        print()
        print("Currently Configured:")
        for config_name, model_name in currently_used.items():
            if model_name:
                print(f"  - {config_name}: {model_name}")


if __name__ == "__main__":
    main()
