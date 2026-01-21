import json
from llm_client import chat

SYSTEM = open("prompts/pwml_system.txt", "r", encoding="utf-8").read()

TEXT = """Glutathione (GSH) is an low-molecular-weight thiol and antioxidant in various species such as plants, mammals and microbes. Glutathione plays important roles in nutrient metabolism, gene expression, etc. and sufficient protein nutrition is important for maintenance of GSH homeostasis. Glutathione is synthesized from glutamate, cysteine, and glycine sequentially by gamma-glutamylcysteine synthetase and GSH synthetase. L-Glutamic acid and cysteine are synthesized to form gamma-glutamylcysteine by glutamate-cysteine ligase that is powered by ATP. Gamma-glutamylcysteine and glycine can be synthesized to form glutathione by enzyme glutathione synthetase that is powered by ATP, too. Glutathione exists oxidized (GSSG) states and in reduced (GSH) state. Oxidation of glutathione happens due to relatively high concentration of glutathione within cells.
"""

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Extract PWML-structured JSON from this text:\n<<<\n{TEXT}\n>>>"}
]

raw = chat(messages, temperature=0, max_tokens=1200)
print("\n=== RAW MODEL OUTPUT ===\n")
print(raw)

print("\n=== PARSED JSON ===\n")
obj = json.loads(raw)
print(json.dumps(obj, indent=2))
