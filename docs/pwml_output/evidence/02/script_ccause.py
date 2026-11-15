import re
def compound_cause(name):
    n=name.strip()
    low=n.lower()
    if re.search(r'\b(mrna|operon|promoter|trna)\b|trna',low) : return 'nucleic_acid_or_tRNA_conjugate'
    if re.search(r'-(entb|ent[a-f])$|^(apo|holo)-|phosphopantetheinyl-',low): return 'protein_bound_or_modified_protein'
    if re.match(r'^(pvd[a-z]|ent[a-f])$',low): return 'protein_name_in_compound_bucket'
    if re.search(r'(s|ids|ines|ols)$',low) and re.search(r'lipid|sterol|sphingo|glycero|phospholipid|oxysterol|bases$',low): return 'class_term'
    if re.search(r'backbone|intermediate$|aglycone$|product$|^prenyl diphosphate$|tetra-acylated|glycosphingolipid',low): return 'class_or_generic_term'
    if re.match(r'^(zn|ca|mg|fe|cu|mn|na|k)[0-9+²⁺ ]*[\+⁺²]*$',low) or re.match(r'^(mgcl2|fe(ii|iii|3|2))$',low) or re.search(r'\+\+|2\+|²⁺|3\+',n) and len(n)<8: return 'ion_or_formula'
    core=re.sub(r'\s*\(.*\)$','',n)
    if len(core)<=9 and re.search(r'[A-Z]{2,}|[0-9][A-Z]',core) and not re.search(r'\s',core): return 'abbreviation'
    if re.match(r'^[A-Za-z0-9\-]{2,8}$',core) and core.upper()==core: return 'abbreviation'
    if re.search(r'\(([A-Za-z0-9\-]{2,8})\)$',n): return 'name_with_abbreviation'
    if re.search(r'lansterol',low): return 'misspelling'
    return 'specific_named_compound'
