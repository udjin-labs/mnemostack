"""Basic embedding example.

Shows how to pick a provider, embed text, and verify dimensions.

Usage:
    # With Gemini (needs GEMINI_API_KEY env var)
    python3 examples/basic_embedding.py --provider gemini

    # With Ollama (needs running Ollama on localhost:11434)
    python3 examples/basic_embedding.py --provider ollama
"""
import argparse

from mnemostack.embeddings import get_provider, list_providers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", default="ollama", choices=list_providers(), help="Embedding provider"
    )
    parser.add_argument("--text", default="What did we decide about authentication?")
    args = parser.parse_args()

    provider = get_provider(args.provider)
    print(f"Provider: {provider.name}")
    print(f"Dimension: {provider.dimension}")

    ok, msg = provider.health_check()
    if not ok:
        print(f"⚠ health check failed: {msg}")
        return

    vec = provider.embed(args.text)
    print(f"Embedded {len(args.text)} chars → {len(vec)}-dim vector")
    print(f"First 5 components: {vec[:5]}")


if __name__ == "__main__":
    main()
