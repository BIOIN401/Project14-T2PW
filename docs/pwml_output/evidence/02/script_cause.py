import re
FAM=re.compile(r"(ases?\b|receptors?\b|channels?\b|ligands?\b|\bfamily\b|\bproteins\b|kinases\b|phosphatases\b|\bPKC\b|\bG protein\b|\bsystem\b|transporters\b|polymerase$|\bfactors?\b$)",re.I)
MOD=re.compile(r"^(apo|holo|phospho|phosphorylated|activated|active|inactive|oxidized|reduced|ubiquitinated|acetylated|methylated)[- ]",re.I)
SRNA=re.compile(r"^(Ryh[A-Z]|Gcv[A-Z]|Spf|OxyS|Hfq$)|\b(sRNA|mRNA|tRNA|rRNA|operon|promoter|gene)\b",re.I)
def protein_cause(name, org=None, kind='protein'):
    n=name.strip()
    if MOD.search(n): return 'modified_form'
    if SRNA.search(n): return 'rna_or_gene_not_protein'
    if re.search(r"[/,]| and |\bcomplex\b|\bsystem\b|-[A-Z]{2,}\d*\b.*\bcomplex", n, re.I) and not re.match(r"^[\w\-\.]+ complex$",n,re.I): return 'complex_or_multi_entity'
    if re.search(r"\bcomplex\b",n,re.I) and re.search(r"synthase|ferrochelatase|chelatase|transport|system|SPT|MCIA|siderosome",n,re.I): return 'complex_or_multi_entity'
    if re.search(r"(ases|receptors|channels|ligands|kinases|phosphatases|polymerases|factors|proteins)$",n,re.I) or re.search(r"\b(receptor|channel|family|subunit)s?$",n,re.I) or re.match(r"^(voltage-gated|G protein|RNA polymerase|sigma)",n,re.I): return 'family_or_class_term'
    # gene-symbol-like
    base=re.sub(r" complex$","",n,flags=re.I)
    if re.match(r"^[A-Za-z]{1,5}[0-9]{0,3}[A-Za-z]?[0-9]{0,2}[a-zA-Z]?(\.\d+)?(-\d+[A-Za-z]?)?$",base) or re.match(r"^[A-Z][a-z]{2}[A-Z]\d?$",base) or re.match(r"^[A-Za-z]+[0-9]+[A-Za-z]*[0-9]*$",base): return 'gene_symbol'
    if re.search(r"\s",base) and re.search(r"ase\b|synthase|reductase|oxidase|transferase|ligase|kinase|dehydrogenase|hydroxylase|cyclase|lyase|mutase|isomerase|deaminase|decarboxylase|hydrolase|phosphatase",base,re.I): return 'descriptive_enzyme_name'
    return 'other'
