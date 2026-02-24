import json
import re
import time


def validate_and_sanitize_json(data):

    if not isinstance(data, dict):
        raise ValueError("Input must be JSON object")

    size = len(json.dumps(data))
    if size > 10000:
        raise ValueError("JSON too large")

    return data


def build_prompt(data, query, docs):

    return f"""
You are an HRMS assistant.

DATA:
{json.dumps(data)}

POLICY DOCS:
{docs}

QUERY:
{query}

Return ONLY JSON array.
If insufficient info return:
[{{"message":"There is no information about this."}}]
"""


def extract_json_from_text(text):

    match = re.search(r"\[.*\]", text, re.S)
    return match.group() if match else text


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start