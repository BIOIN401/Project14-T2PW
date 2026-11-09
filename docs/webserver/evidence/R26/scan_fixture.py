"""Gate before committing the fixture: secret shapes and configured secret values. Prints names and counts only."""
import gzip
import re
import sys
from pathlib import Path

from dotenv import dotenv_values

text = gzip.decompress(Path(sys.argv[1]).read_bytes()).decode("utf-8")
print("size json", len(text))
for name, pat in {"openrouter": r"sk-or-[A-Za-z0-9-]{8,}", "bearer": r"(?i)\bbearer\s+[A-Za-z0-9._-]{8,}",
                  "authorization header": r'(?i)"(proxy-)?authorization"', "cookie header": r'(?i)"(set-)?cookie"',
                  "api key param": r'(?i)\b(api_?key|access_token|token)=(?!<redacted>)[^&\s"]+'}.items():
    print(name, len(re.findall(pat, text)))
emails = sorted(set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}", text)))
print("email addresses:", len(emails))
values = dotenv_values(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.env")
leak = sorted(k for k, v in values.items() if any(s in k for s in ("KEY", "PASSWORD", "USER", "HOST", "EMAIL")) and v and len(v) >= 4 and v in text)
print("configured secret values present:", leak)
