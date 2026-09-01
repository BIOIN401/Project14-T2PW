"""C-108: attack every guard THIS card adds, then restore.

F-144 / D-078. A guard that has not been shown to go RED when it is broken is not
evidence. Each mutation below forces exactly one C-108 guard false and the
focused suites are re-run against it; a mutation that leaves them green means the
guard has no test, not that the guard is fine.

BOTH DIRECTIONS ARE MUTATED, because this card moves the guard both ways:

* N1-N4 break a REFUSAL this card adds -- (a) and (c). They must turn a
  rejection test red.
* N5-N8 break an ADMISSION this card adds or preserves -- (a)'s predications,
  (c)'s over-refusal trap, (c)'s agentive-"by" exemption, (d)'s appositive.
  These matter just as much: C-105 round 1's defect was refusing too much, and a
  preservation battery nobody has mutated is exactly the battery that passed
  while 12 of 29 legitimate cases were refused.
* N9-N11 break the (d) CONTRA in the dangerous direction -- they make the gate
  admit an attenuation aimed at the actor. Merge rule 6 lives here.

N12 IS AN EXPECTED SURVIVOR AND IT IS THE POINT OF MEMBER (b). Reverting
_FOLDED_CHAR_SRC to "[^.]" must leave every BEHAVIOURAL arm green, because the
two spellings are equivalent on folded text -- so it is listed as
``expect="behaviourally-green"``. The ONE arm allowed to go red is the STRUCTURAL
pin that asserts the spelling itself, and that arm is named explicitly below. If
any other arm went red, member (b) would have moved behaviour and would be
mislabelled as code honesty; measured, none does.

RESTORE DISCIPLINE -- D-084. Nothing here hand-rolls a restore. It imports
``apply_mutation`` and ``restore_saved_bytes`` from C-106's repaired
``c102_mutation_attack``, which write bytes throughout, translate the pattern to
the target's newline rather than the target to the pattern's, and prove the
restore by sha256 AND CRLF count. ``git checkout --`` reverts more; a text-mode
round trip reverts less; neither is used, and neither is ``git status
--porcelain``, which a broken restore leaves clean.

Usage::  <python> c108_own_mutations.py <worktree-root>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, restore_saved_bytes, sha256_of,
)

PY = "c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
GUARD = "src/t2pw/curation/apply_audit_patch.py"
TESTS = ["tests/test_c108_f155_class.py",
         "tests/test_c107_actor_cue_calibration.py",
         "tests/test_c105_actor_role_evidence.py"]
BASETEMP = "C:/t/bt/c108mut"

#: The ONLY arm N12 is allowed to turn red: it asserts the SPELLING, not a
#: behaviour, and member (b) is a statement about the spelling.
STRUCTURAL_ONLY = {
    "tests/test_c108_f155_class.py::"
    "test_b_new_the_folded_character_class_is_the_spelling_that_says_so",
}

MUTATIONS = [
    (
        "N1", "red",
        "(a): the transport stem may complete as its agent noun again",
        '        r"transport(?!ers?(?![a-z]))|translocat(?!ors?(?![a-z]))"\n',
        '        r"transport|translocat"  # MUTATION N1\n',
    ),
    (
        "N2", "red",
        "(a): the catalysis stem may complete as 'catalyst' again",
        '        r"catalys(?!ts?(?![a-z]))|catalyz|catalytic|biocatalys(?!ts?(?![a-z]))"\n',
        '        r"catalys|catalyz|catalytic|biocatalys"  # MUTATION N2\n',
    ),
    (
        "N3", "red",
        "(c): the actor-name mask is a no-op, so a name licenses itself again",
        '    folded = _match_fold(actor)\n    if not folded:\n        return window\n',
        '    folded = _match_fold(actor)\n    if True:  # MUTATION N3\n        return window\n',
    ),
    (
        "N4", "red",
        "(a): the agent-noun predication set is emptied, so nothing is re-admitted "
        "-- an OVER-REFUSAL attack on the same fix",
        '        alternatives.append(noun + r"s? (?:for|of|through which|responsible for)")\n',
        '        pass  # MUTATION N4\n',
    ),
    (
        "N5", "red",
        "(c): the mask takes TOKENS instead of contiguous RUNS -- the over-refusal "
        "trap the card names explicitly",
        '        if len(run) < 2 and not (single is not None and run[0][2] == single):\n'
        '            return\n',
        '        pass  # MUTATION N5\n',
    ),
    (
        "N6", "red",
        "(c): the agentive-'by' exemption is switched off, so an agent phrase is "
        "masked as if it were a naming",
        "_AGENTIVE_BY_LOOKBACK = 3\n",
        "_AGENTIVE_BY_LOOKBACK = 0  # MUTATION N6\n",
    ),
    (
        "N7", "red",
        "(c): the cofactor dependence route reads the MASKED window, so the frame "
        "loses the needle it is built around",
        "                dependence is not None and dependence.search(window)\n",
        "                dependence is not None and dependence.search(cue_window)  # MUTATION N7\n",
    ),
    (
        "N8", "red",
        "(d): the contra goes back to being a bare alias of the cue",
        "_CATALYSIS_CONTRA_RE = re.compile(\n"
        "    _C105_INHIBITION_STEMS_CONTRA_SRC + r\"|\" + _C107_INHIBITION_WORDS_SRC\n"
        ")\n",
        '_CATALYSIS_CONTRA_RE = _ROLE_CUE_RES["inhibition"]  # MUTATION N8\n',
    ),
    (
        "N9", "red",
        "(d) ROUND 1, THE POLARITY ITSELF: the agent-noun contra is switched off "
        "on the window route, which is exactly round 0 and exactly REV-108 "
        "blocking finding",
        "            if appositive is not None and _agent_noun_contra_fires(window, appositive):\n"
        "                continue\n"
        "            return True\n"
        "        if family != \"catalysis\":\n",
        "            if False:  # MUTATION N9\n"
        "                continue\n"
        "            return True\n"
        "        if family != \"catalysis\":\n",
    ),
    (
        "N10", "red",
        "(d) ROUND 1: the APPOSITIVE EXEMPTION is removed, so every agent noun "
        "refuses and member (d) itself is lost -- an OVER-REFUSAL attack on the "
        "same repair",
        '                _ATTENUATION_AGENT_NOUN_SRC\n'
        '                + r"(?:\\s+" + _APPOSITIVE_MODIFIER_SRC + r"){0,"\n',
        '                r"(?!)" + _ATTENUATION_AGENT_NOUN_SRC  # MUTATION N10\n'
        '                + r"(?:\\s+" + _APPOSITIVE_MODIFIER_SRC + r"){0,"\n',
    ),
    (
        "N14", "red",
        "(d) ROUND 1: the appositive determiner set gains \"whose\", which is "
        "REV-108 third span -- a relative clause about a DIFFERENT molecule read "
        "as an apposition",
        '_APPOSITIVE_DETERMINER_SRC = r"(?:the|a|an|this|that|its|their)"\n',
        '_APPOSITIVE_DETERMINER_SRC = r"(?:the|a|an|this|that|its|their|whose)"  # MUTATION N14\n',
    ),
    (
        "N15", "red",
        "(d) ROUND 1: the appositive modifier set gains \"bound\", which is "
        "REV-108 fourth and sharpest span -- subject-verb-object read as a noun "
        "phrase, on a span that says the catalysis STOPPED",
        '    r"|bacterial|microbial|human|recombinant)"\n',
        '    r"|bacterial|microbial|human|recombinant|bound)"  # MUTATION N15\n',
    ),
    (
        "N16", "red",
        "(a) R3: the transport VERB inflections are dropped, so dropping the bare "
        "nouns silently drops the verbs too -- the mutation REV-108 found "
        "surviving",
        '        r"|channels|channeled|channelled|channeling|channelling"\n',
        '        r"|(?!)"  # MUTATION N16\n',
    ),
    (
        "N17", "red",
        "(a) R2: the predication modifier gap is closed to zero, so an ordinary "
        "paper modifier breaks the predication -- the other mutation REV-108 "
        "found surviving",
        '            + _PREDICATION_GAP_SRC + r"{0,3}" + noun + r"s?(?![a-z])"\n'
        '        )\n'
        '        # The copular-equivalent verbs.',
        '            + _PREDICATION_GAP_SRC + r"{0,0}" + noun + r"s?(?![a-z])"  # MUTATION N17\n'
        '        )\n'
        '        # The copular-equivalent verbs.',
    ),
    (
        "N18", "red",
        "(a) R-c: the predication gap may cross a genitive again, so "
        "\"P is a substrate of the transporter TonB\" licenses P as a transporter",
        '_PREDICATION_GAP_SRC = r"(?:(?!of )[a-z0-9]+ )"\n',
        '_PREDICATION_GAP_SRC = r"(?:[a-z0-9]+ )"  # MUTATION N18\n',
    ),
    (
        "N11", "red",
        "(d): the contra anchor becomes a WORD anchor, which also stops "
        "'inhibitory' and 'repression' firing",
        '    r"inhibit(?!ors?(?![a-z]))|suppress(?!ors?(?![a-z]))|repress(?!ors?(?![a-z]))"\n',
        '    r"inhibit(?![a-z])|suppress(?![a-z])|repress(?![a-z])"  # MUTATION N11\n',
    ),
    (
        "N13", "red",
        "(d) RE-PINNED FROM C-107 M16. The six inhibition additions revert "
        "to bare unanchored stems. C-107 own M16 keys on the dict-literal "
        "spelling the (d) split replaced, so it now ABORTS with "
        "'matched 0 times' (evidence/c108_c107_mutations_at_tip.log). The "
        "PROPERTY is unchanged and is re-pinned here, at the constant the "
        "stems moved to.",
        '_C107_INHIBITION_WORDS_SRC = (\n'
        '    r"(?<![a-z])(?:blockades?|impair(?:s|ed|ing|ment|ments)?"\n'
        '    r"|silenc(?:e|es|ed|ing)"\n'
        '    r"|sequestr(?:ation|ations|ate|ates|ated|ating)"\n'
        '    r"|ablat(?:e|es|ed|ing|ion|ions)"\n'
        '    r"|interfer(?:e|es|ed|ing|ence))(?![a-z])"\n',
        '_C107_INHIBITION_WORDS_SRC = (  # MUTATION N13\n'
        '    r"blockade|impair|silenc|sequestr|ablat|interfer(?:e|i)"\n',
    ),
    (
        "N12", "behaviourally-green",
        "(b) EXPECTED SURVIVOR: the folded character class reverts to '[^.]'. "
        "Every BEHAVIOURAL arm must stay green; only the structural pin on the "
        "spelling may go red. Any other red would mean member (b) moved "
        "behaviour and was mislabelled as code honesty.",
        '_FOLDED_CHAR_SRC = r"[a-z0-9 ]"\n',
        '_FOLDED_CHAR_SRC = r"[^.]"  # MUTATION N12\n',
    ),
]


def run_suite(root: Path):
    proc = subprocess.run(
        [PY, "-m", "pytest", *TESTS, "-q", "--no-header", "-rf",
         "--basetemp=" + BASETEMP],
        cwd=str(root), capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()
    summary = tail[-1] if tail else "(no output)"
    failed = re.findall(r"^FAILED (\S+)", proc.stdout, re.M)
    return proc.returncode, summary, failed


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    target = root / GUARD
    Path(BASETEMP).mkdir(parents=True, exist_ok=True)

    original = target.read_bytes()
    print("target sha256 before anything:", sha256_of(original))
    print("target CRLF count            :", crlf_count(original))

    code, summary, _f = run_suite(root)
    print()
    print("BASELINE (unmutated): exit=%d  %s" % (code, summary))
    if code != 0:
        print("BASELINE IS NOT GREEN -- every mutation below would be "
              "uninterpretable. Stopping.")
        return 2

    bad = 0
    for name, expect, why, old, new in MUTATIONS:
        print()
        print("=" * 92)
        print("%s [expect %s] %s" % (name, expect.upper(), why))
        print("=" * 92)
        try:
            saved = apply_mutation(target, old, new)
        except ValueError as exc:
            print("  COULD NOT APPLY: %s -- the mutation's source text moved; "
                  "re-pin it before reading on" % exc)
            bad += 1
            continue
        try:
            code, summary, failed = run_suite(root)
        finally:
            restore_saved_bytes(target, saved)
        got = "green" if code == 0 else "red"
        if expect == "behaviourally-green":
            ok = set(failed) <= STRUCTURAL_ONLY
            verdict = ("OK -- every behavioural arm survived; the only red is the "
                       "structural pin on the spelling"
                       if ok else "<< UNEXPECTED -- member (b) moved BEHAVIOUR")
        else:
            ok = got == expect
            verdict = "OK" if ok else "<< UNEXPECTED -- this guard is untested"
        if not ok:
            bad += 1
        print("  result: %s  (exit=%d)  %s" % (got.upper(), code, summary))
        if failed:
            print("  tests red: %s" % ", ".join(sorted(set(failed))[:6]))
        print("  %s" % verdict)

    after = target.read_bytes()
    print()
    print("=" * 92)
    print("target sha256 after everything:", sha256_of(after))
    print("byte-identical to the start   :", sha256_of(after) == sha256_of(original))
    print("CRLF count preserved          :", crlf_count(after) == crlf_count(original))
    print("C108 OWN-MUTATION FAILURES: %d of %d  (must be 0)" % (bad, len(MUTATIONS)))
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
