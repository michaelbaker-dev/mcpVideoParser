#!/usr/bin/env python3
"""Debug argument parsing."""
import argparse
import sys

parser = argparse.ArgumentParser(description="Debug args")
parser.add_argument("--http", action="store_true", help="Run as HTTP server")
parser.add_argument("--host", default="localhost", help="HTTP host")
parser.add_argument("--port", type=int, default=8000, help="HTTP port")

args = parser.parse_args(sys.argv[1:])

print(f"Args: {args}")
print(f"HTTP flag: {args.http}")
print(f"Host: {args.host}")
print(f"Port: {args.port}")