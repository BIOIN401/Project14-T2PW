
def main():
    out = sys.argv[1]
    roots = sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    seen = {}
    rows = []
    skipped = Counter()
    for root in roots:
        for dp, dn, fn in os.walk(root):
            if "final_mapped.json" not in fn:
                continue
            fp = os.path.join(dp, "final_mapped.json").replace("\\", "/")
            if "ABANDONED" in fp:
                skipped["abandoned_dir"] += 1
                continue
            h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            if h in seen:
                skipped["byte_identical_copy"] += 1
                continue
            seen[h] = fp
            q = os.path.join(dp, "quarantine_report.json")
            qok = None
            if os.path.exists(q):
                try:
                    qok = bool(json.load(open(q, encoding="utf-8")).get("ok"))
                except Exception:
                    qok = None
            try:
                r = scan(fp)
            except Exception as ex:
                skipped["unparseable"] += 1
                continue
            r.update(path=fp, quarantine_ok=qok, root=root.replace("\\", "/"))
            pw = [x for x in os.listdir(dp) if x.endswith(".pwml")]
            if pw:
                try:
                    pm = pwml_dup.analyze(os.path.join(dp, pw[0]))
                    keep = ("n_entities", "n_locations", "entities_located", "c_same_state_redraws", "c_entities_with_multi_locations", "d_entities_in_multi_states", "b_name_extra_records", "b_id_extra_records", "n_reactions", "n_transports", "a_reaction_exact_dups", "a_transport_exact_dups")
                    r["pwml"] = {k: pm[k] for k in keep}
                except Exception as ex:
                    r["pwml_error"] = repr(ex)[:100]
            rows.append(r)
    with open(os.path.join(out, "runs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"skipped": dict(skipped), "n_scanned": len(rows)}, open(os.path.join(out, "scan_meta.json"), "w"))
    print("scanned", len(rows), "skipped", dict(skipped))

if __name__ == "__main__":
    main()
