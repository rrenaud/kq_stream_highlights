"""
Run the highlight rater server.

Usage:
    python -m highlight_rater [--port PORT]
"""

import argparse
from .server import run_server


def main():
    parser = argparse.ArgumentParser(description='Run the highlight rating server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=not args.no_debug)


if __name__ == '__main__':
    main()
