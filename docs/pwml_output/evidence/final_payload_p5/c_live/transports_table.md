## PMC11487621
### off_1
Stage 1 (step changed nothing):
  - Pantothenate import via ECF transporter | cargo pantothenate | from `Bacillus subtilis extracellular` | to `Bacillus subtilis cytoplasm` | ok
  - Cysteinopantetheine import via TcyJKLMN | cargo cysteinopantetheine | from `Bacillus subtilis extracellular` | to `Bacillus subtilis cytoplasm` | ok
normalized:
  - Pantothenate import via ECF transporter | cargo pantothenate | from `Bacillus subtilis extracellular` | to `Bacillus subtilis cytoplasm` | ok
  - Cysteinopantetheine import via TcyJKLMN | cargo cysteinopantetheine | from `Bacillus subtilis extracellular` | to `Bacillus subtilis cytoplasm` | ok

### off_2
Stage 1 (step changed nothing):
  (no transports)
normalized:
  (no transports)

### on_1
Stage 1 (step changed nothing):
  - Pantothenate uptake via ECF transporter | cargo pantothenate | from `extracellular state` | to `cytoplasmic state` | ok
  - Cysteinopantetheine uptake via TcyJKLMN | cargo cysteinopantetheine | from `extracellular state` | to `cytoplasmic state` | ok
  - Pantethine uptake | cargo pantethine | from `extracellular state` | to `cytoplasmic state` | ok
normalized:
  - Pantothenate uptake via ECF transporter | cargo pantothenate | from `extracellular state` | to `cytoplasmic state` | ok
  - Cysteinopantetheine uptake via TcyJKLMN | cargo cysteinopantetheine | from `extracellular state` | to `cytoplasmic state` | ok
  - Pantethine uptake | cargo pantethine | from `extracellular state` | to `cytoplasmic state` | ok

### on_2
Stage 1 (step changed nothing):
  - Pantothenate high-affinity uptake via ECF transporter | cargo pantothenate | from `extracellular state` | to `cytoplasmic state` | ok
  - Pantothenate low-affinity uptake | cargo pantothenate | from `extracellular state` | to `cytoplasmic state` | ok
  - Cysteinopantetheine uptake via TcyJKLMN | cargo cysteinopantetheine | from `extracellular state` | to `cytoplasmic state` | ok
normalized:
  - Pantothenate high-affinity uptake via ECF transporter | cargo pantothenate | from `extracellular state` | to `cytoplasmic state` | ok
  - Pantothenate low-affinity uptake | cargo pantothenate | from `extracellular state` | to `cytoplasmic state` | ok
  - Cysteinopantetheine uptake via TcyJKLMN | cargo cysteinopantetheine | from `extracellular state` | to `cytoplasmic state` | ok

## PMC12096016
### off_1
Stage 1 (step changed nothing):
  - enterobactin secretion to extracellular environment | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
normalized:
  - enterobactin secretion to extracellular environment | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok

### off_2
Stage 1 (step changed nothing):
  - Enterobactin secretion via TolC | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
normalized:
  - Enterobactin secretion via TolC | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok

### on_1
Stage 1 (step changed nothing):
  - enterobactin secretion | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
normalized:
  - enterobactin secretion | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok

### on_2
Stage 1 (step changed nothing):
  - Enterobactin secretion via TolC | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
normalized:
  - Enterobactin secretion via TolC | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok

### on_s1
Stage 1 (step changed nothing):
  (no transports)

### on_s2
Stage 1 (step changed nothing):
  - Enterobactin secretion | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
  - Ferric-enterobactin import | cargo ferric-enterobactin | from `extracellular state` | to `cytoplasmic state` | ok

### on_s3
Stage 1 (step changed nothing):
  - Enterobactin secretion via TolC | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok

### on_s4
Stage 1 (step changed nothing):
  - enterobactin secretion | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
  - ferric-enterobactin import | cargo ferric enterobactin | from `extracellular state` | to `cytoplasmic state` | ok

### on_s5
Stage 1 (step changed nothing):
  (no transports)

### on_s6
Stage 1 (step changed nothing):
  - enterobactin secretion | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
  - ferric enterobactin import | cargo ferric enterobactin | from `extracellular state` | to `cytoplasmic state` | ok

### on_s7
Stage 1 (step changed nothing):
  - Enterobactin secretion via TolC | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
  - Ferric-enterobactin import via TonB | cargo ferric-enterobactin | from `extracellular state` | to `cytoplasmic state` | ok

### on_s8
Stage 1 (step changed nothing):
  - Enterobactin secretion | cargo enterobactin | from `cytoplasmic state` | to `extracellular state` | ok
  - Ferric enterobactin import | cargo ferric enterobactin | from `extracellular state` | to `cytoplasmic state` | ok

## PMC12421875
### off_1
Stage 1 (step changed nothing):
  - DHNA secretion | cargo DHNA | from `cytoplasmic state` | to `extracellular state` | ok
  - DHNA diffusion | cargo DHNA | from `inside cell state` | to `extracellular state` | ok
normalized:
  - DHNA secretion | cargo DHNA | from `cytoplasmic state` | to `extracellular state` | ok
  - DHNA diffusion | cargo DHNA | from `inside cell state` | to `extracellular state` | ok

### off_2
Stage 1 (step changed nothing):
  - DHNA diffusion out of cell | cargo DHNA | from `cytoplasmic state` | to `extracellular state` | ok
  - DHNA diffusion from intracellular to extracellular | cargo DHNA | from `intracellular state` | to `extracellular state` | ok
normalized:
  - DHNA diffusion out of cell | cargo DHNA | from `cytoplasmic state` | to `extracellular state` | ok
  - DHNA diffusion from intracellular to extracellular | cargo DHNA | from `intracellular state` | to `extracellular state` | ok

### on_1
Stage 1 (step changed nothing):
  - DHNA secretion | cargo DHNA | from `L. lactis intracellular` | to `extracellular` | ok
  - DHNA diffusion out of cell | cargo DHNA | from `Lactococcus lactis cell` | to `Lactococcus lactis extracellular` | ok
normalized:
  - DHNA secretion | cargo DHNA | from `L. lactis intracellular` | to `extracellular` | ok
  - DHNA diffusion out of cell | cargo DHNA | from `Lactococcus lactis cell` | to `Lactococcus lactis extracellular` | ok

### on_2
Stage 1 (step changed nothing):
  - DHNA diffusion | cargo DHNA | from `L. lactis intracellular state` | to `L. lactis extracellular state` | ok
  - DHNA diffusion | cargo DHNA | from `Lactococcus lactis subsp. lactis KF147 intracellular` | to `Lactococcus lactis subsp. lactis KF147 extracellular` | ok
normalized:
  - DHNA diffusion | cargo DHNA | from `L. lactis intracellular state` | to `L. lactis extracellular state` | ok
  - DHNA diffusion | cargo DHNA | from `Lactococcus lactis subsp. lactis KF147 intracellular` | to `Lactococcus lactis subsp. lactis KF147 extracellular` | ok

## PMC12444477
### off_1
Stage 1 (step changed nothing):
  - MsbA-mediated core-lipid A flip | cargo core-lipid A | from `E. coli inner membrane` | to `E. coli periplasm` | ok
  - Lpt-mediated core-lipid A transport to OM | cargo core-lipid A | from `E. coli periplasm` | to `E. coli outer membrane` | ok
normalized:
  - MsbA-mediated core-lipid A flip | cargo core-lipid A | from `E. coli inner membrane` | to `E. coli periplasm` | ok
  - Lpt-mediated core-lipid A transport to OM | cargo core-lipid A | from `E. coli periplasm` | to `E. coli outer membrane` | ok

### off_2
Stage 1 (step changed nothing):
  - LPS transport to outer membrane | cargo lipopolysaccharide | from `inner membrane state` | to `outer membrane state` | ok
  - Cardiolipin transfer to outer membrane | cargo cardiolipin | from `inner membrane state` | to `outer membrane state` | ok
normalized:
  - LPS transport to outer membrane | cargo lipopolysaccharide | from `inner membrane state` | to `outer membrane state` | ok
  - Cardiolipin transfer to outer membrane | cargo cardiolipin | from `inner membrane state` | to `outer membrane state` | ok

### on_1
Stage 1 (step changed nothing):
  - MsbA-mediated LPS transport | cargo LPS | from `E. coli inner membrane` | to `E. coli outer membrane` | ok
normalized:
  - MsbA-mediated LPS transport | cargo LPS | from `E. coli inner membrane` | to `E. coli outer membrane` | ok

### on_2
Stage 1 (step changed nothing):
  - MsbA-mediated core-lipid A flip | cargo core-lipid A | from `cytoplasmic leaflet of IM` | to `periplasmic leaflet of IM` | ok
  - Lpt-mediated core-lipid A transport to OM | cargo core-lipid A | from `periplasmic leaflet of IM` | to `outer membrane` | ok
normalized:
  - MsbA-mediated core-lipid A flip | cargo core-lipid A | from `cytoplasmic leaflet of IM` | to `periplasmic leaflet of IM` | ok
  - Lpt-mediated core-lipid A transport to OM | cargo core-lipid A | from `periplasmic leaflet of IM` | to `outer membrane` | ok

## PMC6112128
### off_1
Stage 1 (step changed nothing):
  - TBN export by TubI | cargo TBN | from `-` | to `-` | equal/neither
normalized:
  - TBN export by TubI | cargo TBN | from `__auto_state__` | to `__auto_state__` | equal/neither

### off_2
Stage 1 (step changed nothing):
  (no transports)
normalized:
  (no transports)

### on_1
Stage 1 before the step:
  - TubI-mediated TBN export | cargo tubercidin | from `cytoplasmic state` | to `-` | one-sided
Stage 1 after the step:
  - TubI-mediated TBN export | cargo tubercidin | from `cytoplasmic state` | to `extracellular` | ok
normalized:
  - TubI-mediated TBN export | cargo tubercidin | from `cytoplasmic state` | to `extracellular` | ok

### on_2
Stage 1 (step changed nothing):
  - TubI-mediated tubercidin export | cargo tubercidin | from `-` | to `-` | equal/neither
normalized:
  - TubI-mediated tubercidin export | cargo tubercidin | from `__auto_state__` | to `__auto_state__` | equal/neither

### on_s1
Stage 1 (step changed nothing):
  (no transports)
