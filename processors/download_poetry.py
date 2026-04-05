#!/usr/bin/env python3
"""
Download Chinese classical poetry from the chinese-poetry GitHub repository
and convert to JSONL format.

GitHub repo: https://github.com/chinese-poetry/chinese-poetry
Uses GitHub API to list and download files with base64 encoding.
"""

import json
import base64
import urllib.request
import urllib.error
import os
from pathlib import Path

GITHUB_API = "https://api.github.com"
REPO_OWNER = "chinese-poetry"
REPO_NAME = "chinese-poetry"
OUTPUT_DIR = Path(__file__).parent.parent / "collection" / "chinese-poetry"


def github_api_request(path: str) -> dict:
    """Make a request to the GitHub API."""
    url = f"{GITHUB_API}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        raise


def get_file_content(repo_path: str) -> dict:
    """Get file content from GitHub API (returns decoded JSON)."""
    path = f"/repos/{REPO_OWNER}/{REPO_NAME}/contents/{repo_path}"
    data = github_api_request(path)
    if isinstance(data, dict) and "content" in data:
        # File content is base64 encoded
        content = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(content)
    return data


def list_json_files(path: str = "") -> list:
    """List all JSON files in the repository at the given path."""
    path = f"/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    contents = github_api_request(path)
    json_files = []
    for item in contents:
        if item["type"] == "file" and item["name"].endswith(".json"):
            json_files.append(item)
        elif item["type"] == "dir":
            json_files.extend(list_json_files(item["path"]))
    return json_files


def convert_to_jsonl(poetry_data: list, author: str = None) -> list:
    """
    Convert poetry JSON to JSONL format.

    Input format: {'paragraphs': [...], 'author': '...', ...}
    Output format: {'text': 'concatenated paragraphs'}
    """
    records = []
    for poem in poetry_data:
        # Concatenate paragraphs with newline
        paragraphs = poem.get("paragraphs", [])
        text = "\n".join(paragraphs)

        record = {"text": text}

        # Optionally include author if available
        if author:
            record["author"] = author
        elif "author" in poem:
            record["author"] = poem["author"]

        records.append(record)
    return records


def get_collection_files() -> list:
    """Get list of relevant JSON files from the collection directory."""
    return [
        # Tang poetry (poet.tang.*.json)
        "poet.tang.0.json",
        "poet.tang.1.json",
        "poet.tang.2.json",
        "poet.tang.3.json",
        "poet.tang.4.json",
        "poet.tang.5.json",
        # Song poetry (poet.song.*.json)
        "poet.song.0.json",
        "poet.song.1.json",
        "poet.song.2.json",
        "poet.song.3.json",
        "poet.song.4.json",
        # Chu Ci
        "chuci.json",
    ]


def download_and_convert(repo_path: str) -> list:
    """Download a file from GitHub and convert to JSONL records."""
    print(f"Downloading: {repo_path}")
    try:
        data = get_file_content(repo_path)
        return convert_to_jsonl(data)
    except Exception as e:
        print(f"Error downloading {repo_path}: {e}")
        return []


def main():
    """Main function to download poetry and save as JSONL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files_to_download = get_collection_files()

    all_records = []
    for filename in files_to_download:
        records = download_and_convert(filename)
        all_records.extend(records)
        print(f"  -> Got {len(records)} poems from {filename}")

    # Save combined JSONL
    output_path = OUTPUT_DIR / "poetry.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_records)} poems saved to {output_path}")

    # Also save individual files
    for filename in files_to_download:
        records = download_and_convert(filename)
        if records:
            output_file = OUTPUT_DIR / f"{filename}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
