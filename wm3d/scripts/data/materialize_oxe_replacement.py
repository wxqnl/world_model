#!/usr/bin/env python3
"""Compatibility entry point for the former OXE replacement command."""

from scripts.data.materialize_oxe_default import build_templates, main


__all__ = ["build_templates", "main"]


if __name__ == "__main__":
    main()
