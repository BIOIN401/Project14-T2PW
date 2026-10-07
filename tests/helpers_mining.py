"""Deterministic fixtures for the pathway-mining tests.

No API credit is spent anywhere in the mining test suite: the model is a scripted
function, the literature service is a scripted source, and embeddings run in the
offline lexical mode with a temp-directory cache.

The corpus is a *synthetic* jasmonate-flavoured scenario built to exercise the
policy, not to be biochemically authoritative. It deliberately contains:

* two paper-supported reactions that chemically connect (P1, P2);
* a relevant but disconnected paper-supported reaction (P3);
* a later paper supplying a legitimate biochemical bridge (P4, round 3);
* a wrong-species reaction (P5, *Zea mays*);
* a claim with no species evidence (P6: names the target species but cites no
  evidence span for it; since ORCH-745 B3 ``species_not_stated``, not ambiguous);
* two papers that conflict about a reaction attribute (P2 vs P7, enzyme);
* independent support for one reaction from a second paper (P8 repeats P1);
* a truncated model response (P9).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

_EVIDENCE_RE = re.compile(r"\[evidence_id:\s*([A-Za-z0-9_]+)\]")

TARGET_SPECIES = "Arabidopsis thaliana"
TOPIC = "jasmonate biosynthesis"


# ---------------------------------------------------------------------------
# The synthetic corpus
# ---------------------------------------------------------------------------
@dataclass
class FixturePaper:
    key: str
    ident: str
    title: str
    abstract: str
    full_text: str
    year: str = "2024"
    source: str = "fixture"
    uri: str = ""
    #: What an eligibility screen would have read off the paper's own metadata.
    observed_organisms: List[str] = field(default_factory=list)

    def candidate(self) -> Any:
        from t2pw.rag.acquire import CandidatePaper

        return CandidatePaper(
            id=self.ident,
            source=self.source,
            title=self.title,
            abstract=self.abstract,
            full_text=self.full_text,
            source_uri=self.uri or f"https://example.org/{self.ident}",
            year=self.year,
            observed_organisms=list(self.observed_organisms),
        )


PAPERS: Dict[str, FixturePaper] = {
    "P1": FixturePaper(
        key="P1",
        ident="PMC1000001",
        title="Lipoxygenase LOX2 oxygenates alpha-linolenic acid in Arabidopsis thaliana leaves",
        abstract=(
            "In Arabidopsis thaliana leaves, the chloroplastic lipoxygenase LOX2 converts "
            "alpha-linolenic acid to 13-hydroperoxyoctadecatrienoic acid (13-HPOT). "
            "The reaction was measured in planta in Arabidopsis thaliana."
        ),
        full_text=(
            "Results. We assayed leaf extracts of Arabidopsis thaliana. LOX2 converts "
            "alpha-linolenic acid to 13-HPOT in the chloroplast. Activity was abolished in "
            "the lox2 mutant, confirming that LOX2 catalyses this step in Arabidopsis "
            "thaliana."
        ),
    ),
    "P2": FixturePaper(
        key="P2",
        ident="PMC1000002",
        title="Allene oxide synthase converts 13-HPOT to 12-OPDA in Arabidopsis thaliana",
        abstract=(
            "In Arabidopsis thaliana leaves, 13-HPOT is converted to 12-oxophytodienoic acid "
            "(12-OPDA) by allene oxide synthase (AOS) in the chloroplasts."
        ),
        full_text=(
            "Results. In Arabidopsis thaliana leaves, AOS converted 13-HPOT to 12-OPDA. "
            "The product was confirmed by mass spectrometry in Arabidopsis thaliana leaves. "
            "The reaction proceeds in the chloroplast."
        ),
    ),
    "P3": FixturePaper(
        key="P3",
        ident="PMC1000003",
        title="CYP94B3 hydroxylates JA-Ile in Arabidopsis thaliana",
        abstract=(
            "In Arabidopsis thaliana leaves, the cytochrome P450 CYP94B3 converts JA-Ile to "
            "12OH-JA-Ile, attenuating jasmonate signalling."
        ),
        full_text=(
            "Results. In Arabidopsis thaliana, CYP94B3 converts JA-Ile to 12OH-JA-Ile. "
            "Accumulation of 12OH-JA-Ile was lost in cyp94b3 plants of Arabidopsis thaliana."
        ),
    ),
    "P4": FixturePaper(
        key="P4",
        ident="PMC1000004",
        title="From 12-OPDA to JA-Ile: the conjugation route in Arabidopsis thaliana",
        abstract=(
            "In Arabidopsis thaliana leaves, 12-OPDA is converted to jasmonic acid. In "
            "Arabidopsis thaliana leaves, JAR1 converts jasmonic acid to JA-Ile, conjugating it "
            "with L-isoleucine."
        ),
        full_text=(
            "Results. In Arabidopsis thaliana peroxisomes, 12-OPDA is converted to jasmonic "
            "acid. JAR1 of Arabidopsis thaliana then conjugates jasmonic acid with "
            "L-isoleucine to form JA-Ile, as shown by in planta feeding experiments."
        ),
    ),
    "P5": FixturePaper(
        key="P5",
        ident="PMC1000005",
        title="A maize lipoxygenase oxygenates linoleic acid",
        abstract=(
            "In Zea mays, a 9-lipoxygenase converts linoleic acid to 9-HPOD. No Arabidopsis "
            "material was used."
        ),
        full_text=(
            "Results. Zea mays seedlings were assayed. The enzyme converts linoleic acid to "
            "9-HPOD in Zea mays. All experiments used Zea mays tissue."
        ),
    ),
    "P6": FixturePaper(
        key="P6",
        ident="PMC1000006",
        title="OPC-8:0 formation from 12-OPDA",
        abstract=(
            "12-OPDA is reduced to OPC-8:0. The plant material is not specified in the "
            "methods."
        ),
        full_text=(
            "Results. 12-OPDA was reduced to OPC-8:0 in the assay. Arabidopsis thaliana is "
            "mentioned only in the introduction as general background."
        ),
        # The screen read the target species off the paper, but no passage ties it
        # to THIS reaction. A paper-level organism is never the claim's species
        # (ORCH-745 B3): the claim is species_not_stated and held back.
        observed_organisms=[TARGET_SPECIES],
    ),
    "P7": FixturePaper(
        key="P7",
        ident="PMC1000007",
        title="Allene oxide cyclase is responsible for 12-OPDA formation",
        abstract=(
            "We report that in Arabidopsis thaliana leaves allene oxide cyclase, not allene "
            "oxide synthase, converts 13-HPOT to 12-OPDA."
        ),
        full_text=(
            "Results. In Arabidopsis thaliana leaves, allene oxide cyclase converted 13-HPOT "
            "to 12-OPDA. We could not reproduce the assignment to allene oxide synthase in "
            "Arabidopsis thaliana."
        ),
    ),
    "P8": FixturePaper(
        key="P8",
        ident="PMC1000008",
        title="Independent confirmation of alpha-linolenic acid oxygenation in Arabidopsis",
        abstract=(
            "An independent laboratory confirms that alpha-linolenic acid is converted to "
            "13-HPOT in Arabidopsis thaliana leaves."
        ),
        full_text=(
            "Results. Using a different assay in Arabidopsis thaliana, we confirm the "
            "conversion of alpha-linolenic acid to 13-HPOT."
        ),
    ),
    "P9": FixturePaper(
        key="P9",
        ident="PMC1000009",
        title="A long paper the model cannot finish reading",
        abstract="Arabidopsis thaliana jasmonate metabolism, extensively reviewed.",
        full_text=(
            "Results. Many reactions of Arabidopsis thaliana jasmonate metabolism are "
            "described here at length. " * 3
        ),
    ),
}

#: Which papers the scripted literature source returns in which round.
SCHEDULE: Dict[int, List[str]] = {
    1: ["P1", "P2", "P5", "P8"],
    2: ["P3", "P7", "P6", "P9"],
    3: ["P4"],
}


# ---------------------------------------------------------------------------
# Scripted literature source
# ---------------------------------------------------------------------------
class ScriptedPaperSource:
    """Returns the scheduled papers for the current round. Never uses the network.

    The test advances :attr:`round` from the controller's ``round_start``
    progress event, which is how a later paper can supply the bridge.
    """

    name = "scripted"

    def __init__(self, schedule: Optional[Mapping[int, Sequence[str]]] = None) -> None:
        self.schedule = {int(k): list(v) for k, v in (schedule or SCHEDULE).items()}
        self.round = 1
        self.search_calls = 0
        self._served: set = set()

    def search(self, context: Mapping[str, Any], *, sources: Sequence[str],
               max_papers: int) -> List[Any]:  # noqa: ARG002
        self.search_calls += 1
        keys = [k for k in self.schedule.get(self.round, []) if k not in self._served]
        keys = keys[: max(0, int(max_papers))] if max_papers else keys
        self._served.update(keys)
        return [PAPERS[k].candidate() for k in keys]

    def full_text(self, candidate: Any) -> str:
        return str(getattr(candidate, "full_text", "") or "")


# ---------------------------------------------------------------------------
# Scripted model
# ---------------------------------------------------------------------------
def _participant(name: str, evidence: str, kind: str = "metabolite") -> Dict[str, Any]:
    return {"name": name, "entity_type": kind, "evidence_ids": [evidence]}


def _claim(
    *,
    substrates: Sequence[str],
    products: Sequence[str],
    evidence: str,
    enzyme: str = "",
    species: str = TARGET_SPECIES,
    species_cited: bool = True,
    compartment: str = "",
    direction: str = "forward",
    hypothesis: bool = False,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "substrates": [_participant(s, evidence) for s in substrates],
        "products": [_participant(p, evidence) for p in products],
        "evidence_ids": [evidence],
        "direction": direction,
        "direction_evidence_ids": [evidence],
        "hypothesis": hypothesis,
        "pathway_names": [TOPIC],
    }
    if enzyme:
        row["enzyme"] = enzyme
        row["enzyme_evidence_ids"] = [evidence]
    if species:
        row["species"] = species
        row["species_evidence_ids"] = [evidence] if species_cited else []
    if compartment:
        row["compartment"] = compartment
        row["compartment_evidence_ids"] = [evidence]
    return row


def _responses(evidence: str) -> Dict[str, Any]:
    """The scripted claim payload for each paper, keyed by a title fragment."""
    return {
        "LOX2 oxygenates": {
            "claims": [
                _claim(
                    substrates=["alpha-linolenic acid"],
                    products=["13-HPOT"],
                    evidence=evidence,
                    enzyme="LOX2",
                    compartment="chloroplast",
                )
            ]
        },
        "Allene oxide synthase converts": {
            "claims": [
                _claim(
                    substrates=["13-HPOT"],
                    products=["12-OPDA"],
                    evidence=evidence,
                    enzyme="allene oxide synthase",
                    compartment="chloroplast",
                )
            ]
        },
        "CYP94B3 hydroxylates": {
            "claims": [
                _claim(
                    substrates=["JA-Ile"],
                    products=["12OH-JA-Ile"],
                    evidence=evidence,
                    enzyme="CYP94B3",
                )
            ]
        },
        "the conjugation route": {
            "claims": [
                _claim(
                    substrates=["12-OPDA"],
                    products=["jasmonic acid"],
                    evidence=evidence,
                ),
                _claim(
                    substrates=["jasmonic acid", "L-isoleucine"],
                    products=["JA-Ile"],
                    evidence=evidence,
                    enzyme="JAR1",
                ),
            ]
        },
        "maize lipoxygenase": {
            "claims": [
                _claim(
                    substrates=["linoleic acid"],
                    products=["9-HPOD"],
                    evidence=evidence,
                    enzyme="9-lipoxygenase",
                    species="Zea mays",
                )
            ]
        },
        "OPC-8:0 formation": {
            "claims": [
                _claim(
                    substrates=["12-OPDA"],
                    products=["OPC-8:0"],
                    evidence=evidence,
                    species=TARGET_SPECIES,
                    species_cited=False,
                )
            ]
        },
        "Allene oxide cyclase is responsible": {
            "claims": [
                _claim(
                    substrates=["13-HPOT"],
                    products=["12-OPDA"],
                    evidence=evidence,
                    enzyme="allene oxide cyclase",
                    compartment="chloroplast",
                )
            ]
        },
        "Independent confirmation": {
            "claims": [
                _claim(
                    substrates=["alpha-linolenic acid"],
                    products=["13-HPOT"],
                    evidence=evidence,
                )
            ]
        },
    }


class ScriptedModel:
    """A ``chat_detailed``-shaped callable driven by the prompt's paper title.

    It cites only evidence ids that are actually present in the prompt, exactly as
    a well-behaved model would, so the citation check in
    :mod:`t2pw.mining.extraction` is exercised rather than bypassed.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
        user = "".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        ids = _EVIDENCE_RE.findall(user)
        evidence = ids[0] if ids else "ev_missing"
        self.calls.append({"chars": len(user), "evidence": evidence, "kwargs": dict(kwargs)})

        # The truncated-completion case: a JSON object that simply stops.
        if "cannot finish reading" in user:
            return SimpleNamespace(
                text='{"claims": [{"substrates": [{"name": "12-OPDA"',
                diagnostics=SimpleNamespace(
                    finish_reason="length", attempts=1, model="scripted",
                    request_hash="req", response_hash="res", attempt_log=[],
                ),
            )

        for marker, payload in _responses(evidence).items():
            if marker in user:
                return SimpleNamespace(
                    text=json.dumps(payload),
                    diagnostics=SimpleNamespace(
                        finish_reason="stop", attempts=1, model="scripted",
                        request_hash="req", response_hash="res", attempt_log=[],
                    ),
                )
        return SimpleNamespace(
            text='{"claims": []}',
            diagnostics=SimpleNamespace(
                finish_reason="stop", attempts=1, model="scripted",
                request_hash="req", response_hash="res", attempt_log=[],
            ),
        )


def empty_model(*_args: Any, **_kwargs: Any) -> Any:
    """A model that always returns an empty 200."""
    return SimpleNamespace(
        text="",
        diagnostics=SimpleNamespace(
            finish_reason="stop", attempts=1, model="scripted-empty",
            request_hash="", response_hash="", attempt_log=[],
        ),
    )


def truncated_model(*_args: Any, **_kwargs: Any) -> Any:
    """A model that always returns truncated JSON with ``finish_reason=length``."""
    return SimpleNamespace(
        text='{"claims": [{"substrates": [{"name": "x"',
        diagnostics=SimpleNamespace(
            finish_reason="length", attempts=1, model="scripted-truncated",
            request_hash="", response_hash="", attempt_log=[],
        ),
    )


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------
def offline_index(config: Any, tmp_path: Path) -> Any:
    """A :class:`PassageIndex` whose embedder never touches the network."""
    from t2pw.mining.retrieval import PassageIndex
    from t2pw.rag.embed import Embedder

    embedder = Embedder(online=False, cache_path=Path(tmp_path) / "embed_cache.json", dim=64)
    return PassageIndex(config, embedder=embedder)


def make_config(tmp_path: Path, **overrides: Any) -> Any:
    """A mining config pointed at a temp runs directory, offline, memory-backed."""
    from t2pw.mining.config import MiningConfig, StartingInput

    budgets = {
        "max_rounds": 5,
        "max_papers_discovered": 40,
        "max_papers_processed": 20,
        "max_llm_calls": 60,
        "max_cost_usd": 1.0,
        "max_wall_clock_seconds": 600.0,
        "max_frontier_size": 25,
        "stop_after_unproductive_rounds": 2,
    }
    budgets.update(overrides.pop("budgets", {}) or {})
    base: Dict[str, Any] = {
        "budgets": budgets,
        "runs_dir": str(Path(tmp_path) / "runs_mining"),
        "offline": True,
        "screen_papers": False,
        "vector_backend": "memory",
        "enabled_via": "test",
    }
    base.update(overrides)
    return MiningConfig.build(
        target_species=overrides.pop("target_species", TARGET_SPECIES),
        topic=overrides.pop("topic", TOPIC),
        starting_inputs=overrides.pop(
            "starting_inputs",
            [StartingInput(kind="seed_metabolite", raw="alpha-linolenic acid",
                           value="alpha-linolenic acid")],
        ),
        overrides=base,
    )


def build_controller(tmp_path: Path, **overrides: Any) -> Any:
    """A fully wired, fully offline controller plus its scripted source/model."""
    from t2pw.mining.controller import MiningController

    config = make_config(tmp_path, **overrides)
    source = ScriptedPaperSource()
    model = ScriptedModel()
    index = offline_index(config, tmp_path)

    def progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "round_start":
            source.round = int(payload.get("round") or 1)

    controller = MiningController(
        config,
        source=source,
        chat_fn=model,
        index=index,
        progress=progress,
        enabled_override=True,
    )
    return SimpleNamespace(controller=controller, config=config, source=source, model=model,
                           index=index)


def run_fixture(tmp_path: Path, **overrides: Any) -> Any:
    """Run the whole fixture scenario end to end, offline."""
    wired = build_controller(tmp_path, **overrides)
    outcome = wired.controller.run(probe_model=False, probe_embeddings=False)
    return SimpleNamespace(outcome=outcome, **vars(wired))


__all__ = [
    "TARGET_SPECIES",
    "TOPIC",
    "PAPERS",
    "SCHEDULE",
    "FixturePaper",
    "ScriptedPaperSource",
    "ScriptedModel",
    "empty_model",
    "truncated_model",
    "offline_index",
    "make_config",
    "build_controller",
    "run_fixture",
]
