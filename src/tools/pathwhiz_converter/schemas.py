from __future__ import annotations

from typing import Any, Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanWarning(_StrictModel):
    type: str
    message: str


class DeleteSpeciesOp(_StrictModel):
    op: Literal["delete_species"]
    species_id: str
    reason: str


class MergeReactionsOp(_StrictModel):
    op: Literal["merge_reactions"]
    keep_reaction_id: str
    drop_reaction_ids: List[str]
    reason: str


class RenameCompartmentOp(_StrictModel):
    op: Literal["rename_compartment"]
    old_id: str
    new_id: str
    reason: str


class DeleteCompartmentOp(_StrictModel):
    op: Literal["delete_compartment"]
    compartment_id: str
    reason: str


class MoveSpeciesCompartmentOp(_StrictModel):
    op: Literal["move_species_compartment"]
    species_id: str
    new_compartment_id: str
    reason: str


class RenameSpeciesOp(_StrictModel):
    op: Literal["rename_species"]
    species_id: str
    new_name: str
    reason: str


class ReactionParticipant(_StrictModel):
    species_id: str
    stoichiometry: float = 1.0


class AddReactionOp(_StrictModel):
    op: Literal["add_reaction"]
    reaction_id: str
    compartment_id: str
    reactants: List[ReactionParticipant] = Field(default_factory=list)
    products: List[ReactionParticipant] = Field(default_factory=list)
    modifiers: List[str] = Field(default_factory=list)
    reversible: bool = False
    name: str = ""
    reason: str


PatchOp = Annotated[
    Union[
        DeleteSpeciesOp,
        MergeReactionsOp,
        RenameCompartmentOp,
        DeleteCompartmentOp,
        MoveSpeciesCompartmentOp,
        RenameSpeciesOp,
        AddReactionOp,
    ],
    Field(discriminator="op"),
]


class PatchPlan(_StrictModel):
    version: Literal["1.0"] = "1.0"
    objective: str
    ops: List[PatchOp] = Field(default_factory=list)
    warnings: List[PlanWarning] = Field(default_factory=list)


class ConverterReport(_StrictModel):
    mode_used: str
    allow_add_reaction: bool = False
    counts_before: Dict[str, int]
    counts_after: Dict[str, int]
    applied_ops: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=dict)
    llm_rounds: List[Dict[str, Any]] = Field(default_factory=list)
    input_hashes: Dict[str, Any] = Field(default_factory=dict)

