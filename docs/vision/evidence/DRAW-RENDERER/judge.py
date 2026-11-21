"""Vision-LLM judge (DRAW-RENDERER step 6): ranks two drawings of the SAME pathway pairwise against a rubric.
It never moves anything and never sees coordinates; it only says which picture reads better. Each pair is asked in both
orders (position bias); a verdict counts only when both orders agree.
usage: judge.py <tree> <out_dir> <pairs.json>   pairs: [{"id":..., "before": png, "after": png}]
Uses the repo's OpenRouter client (it loads the key itself). Stops when the estimated spend reaches BUDGET."""
import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from PIL import Image  # noqa: E402

from t2pw.llm import client as C  # noqa: E402

MODEL = "google/gemini-3.8-flash"
BUDGET = 2.5  # USD, upper-bound estimate below; the brief caps the lane at $3
PRICE_IN, PRICE_OUT = 1.0 / 1e6, 5.0 / 1e6  # deliberately pessimistic per-token prices for the estimate
RUBRIC = """You compare two automatically drawn diagrams, A and B, of the SAME metabolic pathway (same molecules, same steps).
Judge ONLY readability of the drawing, the way a curator of hand-drawn PathBank/PathWhiz maps would:
1 flow: the main chain reads in one direction (wrapping to a new row is fine), little backtracking
2 clutter: few line crossings, no line running through a box or a label, no canvas-wide lines
3 compartments: regions and membranes are clear; transports visibly cross a membrane into the next region
4 conventions: enzymes beside their step; cofactors / common molecules as short side stubs or local copies
5 space: compact, near 1.5-2.5:1, labels readable at page width, little empty space
Do not reward or punish content: both show the same data. Return ONLY JSON:
{"winner": "A" | "B" | "tie", "confidence": 1-5, "criteria": {"flow": "A|B|tie", "clutter": "A|B|tie", "compartments": "A|B|tie",
 "conventions": "A|B|tie", "space": "A|B|tie"}, "why": "one sentence naming visible labels"}"""


def img_url(path, max_side=1800):
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(a_png, b_png):
    msgs = [{"role": "user", "content": [{"type": "text", "text": RUBRIC}, {"type": "text", "text": "Diagram A:"},
                                         {"type": "image_url", "image_url": {"url": img_url(a_png)}},
                                         {"type": "text", "text": "Diagram B:"},
                                         {"type": "image_url", "image_url": {"url": img_url(b_png)}}]}]
    b0 = dict(C._token_stats)
    try:
        res = C.chat_detailed(msgs, temperature=0.0, max_tokens=1500, response_json=True, model_override=MODEL,
                              stage_name="draw_judge", timeout=180)
        out = json.loads(res.text)
    except Exception as e:  # noqa: BLE001
        out = {"error": "%s: %s" % (type(e).__name__, str(e)[:900])}
    b1 = dict(C._token_stats)
    cost = (b1["prompt_tokens"] - b0["prompt_tokens"]) * PRICE_IN + (b1["completion_tokens"] - b0["completion_tokens"]) * PRICE_OUT
    return out, cost


def main():
    out_dir, pairs = sys.argv[2], json.load(open(sys.argv[3], encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    spent, rows = 0.0, []
    for p in pairs:
        if spent >= BUDGET:
            print("BUDGET reached, stopping", flush=True)
            break
        t0 = time.time()
        r1, c1 = ask(p["before"], p["after"])  # A = before
        r2, c2 = ask(p["after"], p["before"])  # A = after
        spent += c1 + c2
        v1 = {"A": "before", "B": "after"}.get(r1.get("winner"), "tie")
        v2 = {"A": "after", "B": "before"}.get(r2.get("winner"), "tie")
        verdict = v1 if v1 == v2 else "inconsistent"
        rows.append(dict(id=p["id"], verdict=verdict, ab=r1, ba=r2, secs=round(time.time() - t0, 1)))
        print(p["id"], verdict, round(spent, 3), flush=True)
    summ = {"model": MODEL, "pairs": len(rows), "after_wins": sum(r["verdict"] == "after" for r in rows),
            "before_wins": sum(r["verdict"] == "before" for r in rows), "ties": sum(r["verdict"] == "tie" for r in rows),
            "inconsistent": sum(r["verdict"] == "inconsistent" for r in rows), "est_spend_usd_upper": round(spent, 3),
            "tokens": dict(C._token_stats)}
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(out_dir, "judge.json"), "w", encoding="utf-8"), indent=1)
    print("JUDGE", json.dumps(summ))


main()
