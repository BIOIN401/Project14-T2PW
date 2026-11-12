# PathWhiz importer bug 3: `transport-transporter-id` not remapped (`lib/pwml_parser.rb`)

**Symptom.** In an empty database, `transport_transporter_visualization: Transport transporter can't be blank` (19 messages in 12 files with the payload flag on). Also occurs on the real PathBank file PW176888.

**Cause.** `add_content` remaps `reaction-enzyme-id` via `@reaction_enzyme_list` (filled by `store_reaction_enzyme_ids`), but has no branch for `transport-transporter-id`. It falls to the generic branch and keeps the raw XML id, which matches no `transport_transporters` row, so the visualization's `transport_transporter` is nil.

**Fix** (`pathwhiz_importer_fixes_p3.patch`, one file, 4 hunks of additions only): new `@transport_transporter_list` (initialised in both places `@reaction_enzyme_list` is); a `transport-transporter-id` branch in `add_content`; and `store_transport_transporter_ids(node, transport)`, called after a `Transport` is saved or found existing. It maps the XML `<transport-transporter><id>` to the created row through `Transport#get_enzyme(protein_complex_id)`, where the protein complex id is first translated through `@protein_complex_list`. This is the same pattern as reaction enzymes.

**Production impact.** Without the fix, an unmapped id keeps the raw XML number. If that number equals a real `transport_transporters.id`, the visualization silently attaches to an unrelated transporter, with no error. With the fix, the id resolves to the transporter created or found for that transport.

**Verification.** `ruby -c` Syntax OK (image `smpdb-app:latest`) on the harness copy and the owner copy. NOT run: an import of the 12 affected files; the harness was not re-run. Not covered: `reaction-coupled-transport` enzymes use the same raw-id path in `store_nested_object` (only `Reaction` calls the id store); not checked.
Applied to: `C:/t/pwi/smpdb`, harness work copy, owner copy (backup `C:/t/pwi/backup/pwml_parser.rb.owner-2026-11-12`; diff vs backup is exactly this hunk set).
