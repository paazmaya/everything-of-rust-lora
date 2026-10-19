#!/usr/bin/env python3
"""Evaluate Rust version knowledge or rescore saved answers without model loading."""

import logging

from lib.version_eval import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
