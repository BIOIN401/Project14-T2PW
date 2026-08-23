"""The payload keys that no LLM PROMPT in this package may carry.

C-079. This constant was introduced by C-075 in :mod:`t2pw.curation.audit_json_llm`
and lived there, because at the time exactly one serializer read it. F-105 then
measured a SECOND serializer with the same job --
:func:`t2pw.curation.interactive_curator.strip_payload_for_interactive_context` --
filtering the other way round, by a blacklist that has to be remembered, and
therefore silently admitting the source index that the audit side already drops.

WHY THE CONSTANT MOVED HERE INSTEAD OF BEING IMPORTED FROM ``audit_json_llm``.
Not an import cycle -- there is none. A layering rule: ``audit_json_llm`` imports
``t2pw.llm.client`` AT MODULE SCOPE, and that module constructs an ``OpenAI``
client while it is being imported and raises ``RuntimeError`` outright when
``LLM_PROVIDER`` is neither ``local`` nor ``openrouter``, or when an OpenRouter
key is missing or malformed (``llm/client.py:71-106``). ``interactive_curator``
is deliberately free of that dependency: it defers ``from t2pw.llm.client import
chat`` into :func:`~t2pw.curation.interactive_curator.run_interactive_curator_round`
so the module -- and the pure payload-shaping functions in it -- stay importable
with no provider configured. Importing ``audit_json_llm`` for one frozenset would
have undone that deliberate choice and made a pure string filter unimportable
without LLM credentials.

So the constant sits in a leaf that costs nothing to import: this module pulls in
:mod:`t2pw.mapping.identity_admission`, whose own imports are ``re`` and
``typing``. ``audit_json_llm`` re-exports the name, so
``audit_json_llm.PROMPT_OMITTED_PAYLOAD_KEYS`` is unchanged for every existing
caller AND is the same object both serializers read -- which is the property that
stops the two sites drifting apart again.
"""

from __future__ import annotations

from t2pw.mapping import identity_admission


#: C-075. Payload keys an LLM PROMPT never needs, removed from the SERIALIZED
#: COPY only. Nothing here is removed from the payload itself: the index has to
#: survive the audit round trip, because the AUTHORITATIVE mapping run happens
#: AFTER the audit (``streamlit_app.py:4225``) and a payload that reached it
#: without an index would fail open and withhold nothing.
#:
#: WHY IT EXISTS. Arming C-075's source-support pass puts a normalized copy of
#: the whole paper on ``final_payload`` at the Stage-2 merge, and the audit loop
#: json.dumps THE ENTIRE PAYLOAD into its user message
#: (:func:`t2pw.curation.audit_json_llm._build_llm_prompt`). Measured over the
#: committed corpus that blob is ~56 KB per leg -- mean 56,404 bytes against a
#: mean source of 58,320 -- so every audit round of every leg would carry it,
#: risking prompt truncation and degraded audit quality across the whole run.
#: The auditor has no use for it: it judges STRUCTURE, and the paper is not part
#: of the structure it patches.
#:
#: C-079 / F-105: the same is true of the INTERACTIVE curator, which json.dumps
#: its working payload into a multimodal prompt every round
#: (``interactive_curator.py:255-259``). F-105 measured the index blob at 64,880
#: bytes on a real leg. That site now reads THIS constant too, so a key added
#: here is dropped from both prompts and neither can silently readmit it.
#:
#: PATCH POINTERS ARE UNAFFECTED. RFC6901 pointers address object members by
#: NAME, not by position, so dropping a top-level key shifts no pointer into
#: ``entities`` or ``processes``. It also removes the possibility of the model
#: proposing a patch AGAINST the index, which it could do while it could see it.
#:
#: Key-driven and additive: a payload carrying none of these keys is serialized
#: by the same object it always was, so no existing prompt changes by one byte.
PROMPT_OMITTED_PAYLOAD_KEYS: frozenset = frozenset({
    identity_admission.SOURCE_INDEX_KEY,
})
