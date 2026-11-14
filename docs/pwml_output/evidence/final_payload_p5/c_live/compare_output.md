## PMC11487621

| run | wall s | LLM requests | deficient | repairs | repair calls | S1 pre==post | transports ok/one-sided/equal (S1) | (normalized) | reactions | entities | interactions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off_1 | 301.6 | 7 | - | - | - | True | 2/0/0 | 2/0/0 | 7 | 47 | 4 |
| off_2 | 171.2 | 7 | - | - | - | True | 0/0/0 | 0/0/0 | 7 | 36 | 1 |
| on_1 | 157.7 | 7 | 0 | 0 | 0 | True | 3/0/0 | 3/0/0 | 12 | 49 | 4 |
| on_2 | 260.1 | 7 | 0 | 0 | 0 | True | 3/0/0 | 3/0/0 | 10 | 51 | 5 |

paired Stage-1 diff on_1 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_2 (only-in-pre / only-in-post): no fact differs (byte-identical)

| pair | kind | reaction | transport | interaction | state | location | entities |
|---|---|---|---|---|---|---|---|
| off_1 vs off_2 | off-off | 1/1 | 2/0 | 4/1 | 3/1 | 16/13 | 15/4 |
| off_1 vs on_1 | off-on | 0/5 | 2/3 | 4/4 | 2/3 | 16/14 | 6/8 |
| off_1 vs on_2 | off-on | 0/3 | 2/2 | 1/2 | 4/3 | 16/18 | 5/9 |
| off_2 vs on_1 | off-on | 0/5 | 0/3 | 1/4 | 1/4 | 13/14 | 1/14 |
| off_2 vs on_2 | off-on | 1/4 | 0/2 | 0/4 | 2/3 | 13/18 | 2/17 |
| on_1 vs on_2 | on-on | 2/0 | 1/0 | 4/5 | 2/0 | 6/10 | 5/7 |

(cells: only-in-first / only-in-second)

## PMC12096016

| run | wall s | LLM requests | deficient | repairs | repair calls | S1 pre==post | transports ok/one-sided/equal (S1) | (normalized) | reactions | entities | interactions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off_1 | 178.6 | 4 | - | - | - | True | 1/0/0 | 1/0/0 | 6 | 24 | 3 |
| off_2 | 127.7 | 4 | - | - | - | True | 1/0/0 | 1/0/0 | 5 | 22 | 3 |
| on_1 | 52.1 | 4 | 0 | 0 | 0 | True | 1/0/0 | 1/0/0 | 5 | 25 | 10 |
| on_2 | 116.7 | 4 | 0 | 0 | 0 | True | 1/0/0 | 1/0/0 | 5 | 19 | 6 |
| on_s1 | 140.6 | 2 | 0 | 0 | 0 | True | 0/0/0 | - | - | - | - |
| on_s2 | 56.2 | 2 | 0 | 0 | 0 | True | 2/0/0 | - | - | - | - |
| on_s3 | 70.0 | 2 | 0 | 0 | 0 | True | 1/0/0 | - | - | - | - |
| on_s4 | 56.4 | 2 | 0 | 0 | 0 | True | 2/0/0 | - | - | - | - |
| on_s5 | 266.5 | 2 | 0 | 0 | 0 | True | 0/0/0 | - | - | - | - |
| on_s6 | 19.1 | 2 | 0 | 0 | 0 | True | 2/0/0 | - | - | - | - |
| on_s7 | 78.4 | 2 | 0 | 0 | 0 | True | 2/0/0 | - | - | - | - |
| on_s8 | 314.2 | 2 | 0 | 0 | 0 | True | 2/0/0 | - | - | - | - |

paired Stage-1 diff on_1 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_2 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s1 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s2 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s3 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s4 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s5 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s6 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s7 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s8 (only-in-pre / only-in-post): no fact differs (byte-identical)

| pair | kind | reaction | transport | interaction | state | location | entities |
|---|---|---|---|---|---|---|---|
| off_1 vs off_2 | off-off | 3/2 | 0/0 | 3/3 | 0/0 | 11/2 | 4/2 |
| off_1 vs on_1 | off-on | 5/4 | 0/0 | 2/9 | 0/0 | 11/0 | 3/4 |
| off_1 vs on_2 | off-on | 4/3 | 0/0 | 3/6 | 1/1 | 5/0 | 8/3 |
| off_2 vs on_1 | off-on | 3/3 | 0/0 | 3/10 | 0/0 | 2/0 | 4/7 |
| off_2 vs on_2 | off-on | 2/2 | 0/0 | 2/5 | 1/1 | 2/6 | 5/2 |
| on_1 vs on_2 | on-on | 4/4 | 0/0 | 10/6 | 1/1 | 0/6 | 11/5 |

(cells: only-in-first / only-in-second)

## PMC12421875

| run | wall s | LLM requests | deficient | repairs | repair calls | S1 pre==post | transports ok/one-sided/equal (S1) | (normalized) | reactions | entities | interactions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off_1 | 153.0 | 7 | - | - | - | True | 2/0/0 | 2/0/0 | 10 | 26 | 1 |
| off_2 | 320.3 | 7 | - | - | - | True | 2/0/0 | 2/0/0 | 9 | 25 | 1 |
| on_1 | 392.5 | 7 | 0 | 0 | 0 | True | 2/0/0 | 2/0/0 | 9 | 24 | 1 |
| on_2 | 558.1 | 7 | 0 | 0 | 0 | True | 2/0/0 | 2/0/0 | 10 | 25 | 2 |

paired Stage-1 diff on_1 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_2 (only-in-pre / only-in-post): no fact differs (byte-identical)

| pair | kind | reaction | transport | interaction | state | location | entities |
|---|---|---|---|---|---|---|---|
| off_1 vs off_2 | off-off | 3/2 | 1/1 | 1/1 | 2/1 | 6/3 | 5/4 |
| off_1 vs on_1 | off-on | 3/2 | 2/2 | 0/0 | 4/4 | 25/14 | 5/3 |
| off_1 vs on_2 | off-on | 3/3 | 2/2 | 0/1 | 4/4 | 25/15 | 5/4 |
| off_2 vs on_1 | off-on | 0/0 | 2/2 | 1/1 | 3/4 | 22/14 | 2/1 |
| off_2 vs on_2 | off-on | 0/1 | 2/2 | 0/1 | 3/4 | 22/15 | 1/1 |
| on_1 vs on_2 | on-on | 0/1 | 2/2 | 0/1 | 4/4 | 14/15 | 0/1 |

(cells: only-in-first / only-in-second)

## PMC12444477

| run | wall s | LLM requests | deficient | repairs | repair calls | S1 pre==post | transports ok/one-sided/equal (S1) | (normalized) | reactions | entities | interactions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off_1 | 148.9 | 7 | - | - | - | True | 2/0/0 | 2/0/0 | 11 | 64 | 15 |
| off_2 | 875.3 | 7 | - | - | - | True | 2/0/0 | 2/0/0 | 14 | 56 | 17 |
| on_1 | 669.9 | 7 | 0 | 0 | 0 | True | 1/0/0 | 1/0/0 | 19 | 72 | 18 |
| on_2 | 820.3 | 8 | 0 | 0 | 0 | True | 2/0/0 | 2/0/0 | 8 | 48 | 12 |

paired Stage-1 diff on_1 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_2 (only-in-pre / only-in-post): no fact differs (byte-identical)

| pair | kind | reaction | transport | interaction | state | location | entities |
|---|---|---|---|---|---|---|---|
| off_1 vs off_2 | off-off | 8/11 | 2/2 | 13/15 | 3/4 | 35/18 | 29/21 |
| off_1 vs on_1 | off-on | 11/19 | 2/1 | 13/16 | 0/4 | 26/42 | 17/25 |
| off_1 vs on_2 | off-on | 11/8 | 2/2 | 13/10 | 4/6 | 39/32 | 25/9 |
| off_2 vs on_1 | off-on | 12/17 | 2/1 | 12/13 | 0/3 | 13/46 | 16/32 |
| off_2 vs on_2 | off-on | 14/8 | 2/2 | 15/10 | 5/6 | 22/32 | 23/15 |
| on_1 vs on_2 | on-on | 18/7 | 1/2 | 15/9 | 8/6 | 55/32 | 30/6 |

(cells: only-in-first / only-in-second)

## PMC6112128

| run | wall s | LLM requests | deficient | repairs | repair calls | S1 pre==post | transports ok/one-sided/equal (S1) | (normalized) | reactions | entities | interactions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off_1 | 251.1 | 4 | - | - | - | True | 0/0/1 | 0/0/1 | 8 | 31 | 0 |
| off_2 | 331.3 | 4 | - | - | - | True | 0/0/0 | 0/0/0 | 9 | 26 | 1 |
| on_1 | 351.7 | 5 | 1 | 1 | 1 | False | 1/0/0 | 1/0/0 | 9 | 33 | 0 |
| on_2 | 856.9 | 6 | 1 | 0 | 1 | True | 0/0/1 | 0/0/1 | 10 | 37 | 0 |
| on_s1 | 61.3 | 2 | 0 | 0 | 0 | True | 0/0/0 | - | - | - | - |

### PMC6112128 on_1: repair record

```json
{
 "calls": 1,
 "deficient": [
  {
   "name": "TubI-mediated TBN export",
   "pointer": "/processes/transports/0",
   "sides": [
    "cytoplasmic state",
    ""
   ]
  }
 ],
 "error": null,
 "flag": "T2PW_TRANSPORT_SIDES",
 "new_states": [
  {
   "pointer": "/processes/transports/0",
   "state": {
    "evidence": "out of the cell",
    "name": "extracellular",
    "species": "Streptomyces tubercidicus NBRC 13090",
    "subcellular_location": "extracellular"
   }
  }
 ],
 "rejected": [],
 "repairs": [
  {
   "name": "TubI-mediated TBN export",
   "new": [
    "cytoplasmic state",
    "extracellular"
   ],
   "old": [
    "cytoplasmic state",
    ""
   ],
   "pointer": "/processes/transports/0",
   "quotes": {
    "to_biological_state": "out of the cell"
   }
  }
 ]
}
```

### PMC6112128 on_2: repair record

```json
{
 "calls": 1,
 "deficient": [
  {
   "name": "TubI-mediated tubercidin export",
   "pointer": "/processes/transports/0",
   "sides": [
    "",
    ""
   ]
  }
 ],
 "error": null,
 "flag": "T2PW_TRANSPORT_SIDES",
 "new_states": [],
 "rejected": [
  {
   "name": "TubI-mediated tubercidin export",
   "pointer": "/processes/transports/0",
   "reason": "no side with a quote found in the paper text"
  }
 ],
 "repairs": []
}
```

paired Stage-1 diff on_1 (only-in-pre / only-in-post): {"state": [[], ["('extracellular', 'extracellular')"], 1, 2], "transport": [["('tubercidin', 'cytoplasmic state', '')"], ["('tubercidin', 'cytoplasmic state', 'extracellular')"], 1, 1]}
paired Stage-1 diff on_2 (only-in-pre / only-in-post): no fact differs (byte-identical)
paired Stage-1 diff on_s1 (only-in-pre / only-in-post): no fact differs (byte-identical)

| pair | kind | reaction | transport | interaction | state | location | entities |
|---|---|---|---|---|---|---|---|
| off_1 vs off_2 | off-off | 1/2 | 1/0 | 0/1 | 1/1 | 28/0 | 6/1 |
| off_1 vs on_1 | off-on | 3/4 | 1/1 | 0/0 | 1/2 | 28/0 | 5/7 |
| off_1 vs on_2 | off-on | 3/5 | 1/1 | 0/0 | 1/0 | 28/0 | 4/10 |
| off_2 vs on_1 | off-on | 4/4 | 0/1 | 1/0 | 1/2 | 0/0 | 6/13 |
| off_2 vs on_2 | off-on | 4/5 | 0/1 | 1/0 | 1/0 | 0/0 | 3/14 |
| on_1 vs on_2 | on-on | 0/1 | 1/1 | 0/0 | 2/0 | 0/0 | 2/6 |

(cells: only-in-first / only-in-second)
