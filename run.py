#!/usr/bin/env python3
"""
TDS Data Analyst Agent - Entry Point
Single command deployment for Render
"""

import os
import sys
import glob
import mimetypes
import requests
import uvicorn
from app import app

if __name__ == "__main__":
    # CLI mode: if a URL is passed, post questions + data files and print JSON
    if len(sys.argv) >= 2 and sys.argv[1].startswith("http"):
        url = sys.argv[1]
        cwd = os.getcwd()
        # Collect files: questions.txt (required) and data files
        question_path = os.path.join(cwd, "questions.txt")
        if not os.path.exists(question_path):
            # Try to find questions.txt recursively
            matches = glob.glob(os.path.join(cwd, "**", "questions.txt"), recursive=True)
            if matches:
                question_path = matches[0]
        files = []
        opened = []
        try:
            if os.path.exists(question_path):
                fh = open(question_path, "rb")
                opened.append(fh)
                files.append(("questions.txt", ("questions.txt", fh, "text/plain")))
            # Add common data files in the working directory
            patterns = ["*.csv", "*.xlsx", "*.xls", "*.json", "*.parquet", "*.zip", "*.tar.gz", "*.tgz"]
            data_paths = set()
            for pat in patterns:
                for p in glob.glob(os.path.join(cwd, pat)):
                    if os.path.basename(p) == "questions.txt":
                        continue
                    data_paths.add(p)
            # Also search one level deeper for data files
            for pat in patterns:
                for p in glob.glob(os.path.join(cwd, "**", pat), recursive=True):
                    if os.path.basename(p) == "questions.txt":
                        continue
                    data_paths.add(p)
            for p in sorted(data_paths):
                mime, _ = mimetypes.guess_type(p)
                if mime is None:
                    mime = "application/octet-stream"
                fh = open(p, "rb")
                opened.append(fh)
                field_name = os.path.basename(p)
                files.append((field_name, (field_name, fh, mime)))
            resp = requests.post(url, files=files, timeout=120)
            # Print raw text so harness can JSON.parse(output)
            print(resp.text)
        except Exception as e:
            print("{}".format("{}" if False else "{}"))  # print empty JSON string if something goes wrong
            # Fallback minimal JSON object
            print("{}")
        finally:
            for fh in opened:
                try:
                    fh.close()
                except Exception:
                    pass
    else:
        # Get port from environment (Render sets this)
        port = int(os.environ.get("PORT", 8000))
        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
