"""Read-only: does the legacy URL PathWhiz's UniProtBlast.get_uniprot uses (lib/uniprot_blast.rb:61,91) still answer?"""
import sys, urllib.request
for url in ["https://www.uniprot.org/uniprot/P0ADI4.xml", "https://www.uniprot.org/uniprot/P0ADI4.fasta"]:
    req = urllib.request.Request(url, headers={"User-Agent": "orch761-readonly-probe"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read(400).decode("utf-8", "replace")
            print(url, "->", r.status, r.geturl(), r.headers.get("Content-Type"), "|", body[:160].replace("\n", " "))
    except Exception as e:
        print(url, "-> ERROR", e)
