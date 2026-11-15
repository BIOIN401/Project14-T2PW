"""Field inventory per stage/bucket over the canonical runs. Stdlib only. Read-only on run folders."""
import json, os, collections, sys

L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
Q = [x for x in L if x['q'] is True]          # 152 canonical
QIR = [x for x in Q if x['ir']]               # 62 with IR + pwml

BLOB = {'provenance_lineage', 'mapping_meta', 'db_match', 'species_ref', 'original_row', 'original_reaction'}


def tname(v):
    return type(v).__name__


def walk(items, inv, nruns, runid, prefix=''):
    """items: list of dicts (or other). inv: field -> [count, types Counter, runs set]"""
    for it in items:
        if not isinstance(it, dict):
            f = prefix + '<item:' + tname(it) + '>'
            inv[f][0] += 1; inv[f][1][tname(it)] += 1; inv[f][2].add(runid)
            continue
        for k, v in it.items():
            f = prefix + k
            inv[f][0] += 1; inv[f][1][tname(v)] += 1; inv[f][2].add(runid)
            if k in BLOB:
                continue
            if isinstance(v, dict) and k in ('mapped_ids', 'left', 'right', 'withheld', 'species'):
                walk([v], inv, nruns, runid, f + '.')
            if isinstance(v, list) and v and k in ('components', 'enzymes', 'modifiers', 'element_states', 'left', 'right',
                                                   'inputs', 'outputs', 'members', 'transport_elements', 'transporters', 'transported', 'cargo'):
                walk(v, inv, nruns, runid, f + '[].')


def bucket_payload(d):
    B = {}
    ent = d.get('entities') or {}
    for k, v in ent.items():
        if isinstance(v, list):
            B['entities.' + k] = v
    B['biological_states'] = d.get('biological_states') or []
    el = d.get('element_locations') or {}
    for k, v in el.items():
        if isinstance(v, list):
            B['element_locations.' + k] = v
    pr = d.get('processes') or {}
    for k, v in pr.items():
        if isinstance(v, list):
            B['processes.' + k] = v
    return B


def bucket_ir(d):
    B = {}
    for k in ('species', 'subcellular_locations', 'cell_types', 'tissues', 'biological_states', 'locations',
              'protein_complex_visualizations', 'bound_visualizations', 'edges', 'process_visualizations'):
        v = d.get(k)
        if isinstance(v, list):
            B[k] = v
    for k, v in (d.get('entities') or {}).items():
        if isinstance(v, list):
            B['entities.' + k] = v
    for k, v in (d.get('processes') or {}).items():
        if isinstance(v, list):
            B['processes.' + k] = v
    B['pathway'] = [d['pathway']] if isinstance(d.get('pathway'), dict) else []
    return B


def run(stage, fname, bucketer, runs):
    inv = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, collections.Counter(), set()]))
    cnt = collections.Counter(); top = collections.Counter()
    nok = 0
    for x in runs:
        p = os.path.join(x['dir'], fname)
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        nok += 1
        for k in d.keys():
            top[k] += 1
        for b, items in bucketer(d).items():
            cnt[b] += len(items)
            walk(items, inv[b], len(runs), x['dir'])
    return inv, cnt, top, nok


if __name__ == '__main__':
    out = {}
    for stage, fname, bk, runs in (('merged_payload', 'merged_payload.json', bucket_payload, Q),
                                   ('final_mapped', 'final_mapped.json', bucket_payload, Q),
                                   ('pwml_ir', 'pwml_ir.json', bucket_ir, QIR),
                                   ('merged_payload_62', 'merged_payload.json', bucket_payload, QIR),
                                   ('final_mapped_62', 'final_mapped.json', bucket_payload, QIR)):
        inv, cnt, top, nok = run(stage, fname, bk, runs)
        S = {'runs_read': nok, 'top_level': dict(top), 'buckets': {}}
        for b in sorted(inv):
            S['buckets'][b] = {'n_items': cnt[b], 'fields': {
                f: {'n': v[0], 'frac': round(v[0] / cnt[b], 3) if cnt[b] else None, 'types': dict(v[1]), 'runs': len(v[2])}
                for f, v in sorted(inv[b].items())}}
        out[stage] = S
        print(stage, nok, {b: cnt[b] for b in sorted(cnt)})
    json.dump(out, open('C:/t/ev-oa/payload/inventory.json', 'w'), indent=1)
