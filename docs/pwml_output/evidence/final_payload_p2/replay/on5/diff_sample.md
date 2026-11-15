# Diff sample (flag-off vs flag-on), 5 of 97 differing PWML

## idx 78 (P0), ids written: 17, title: The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB.

```diff
--- flag-off
+++ flag-on
@@ -3,5 +3,5 @@
   <named-for-id type="integer">1</named-for-id>
   <named-for-type>Pathway</named-for-type>
-  <cached-name>Generated Pathway</cached-name>
+  <cached-name>The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB.</cached-name>
   <cached-description></cached-description>
   <cached-subject>Metabolic</cached-subject>
@@ -18,10 +18,18 @@
         <pathway>
           <id type="integer">1</id>
-          <name>Generated Pathway</name>
+          <name>The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB.</name>
           <description></description>
           <subject>Metabolic</subject>
           <species-id type="integer">3</species-id>
           <sub-pathways/>
-          <references/>
+          <references>
+            <reference>
+              <id type="integer">1</id>
+              <pubmed-id>40400396</pubmed-id>
+              <text>Bin X, Pawelek PD: The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB. 2025. doi: 10.1002/pro.70160. PMID: 40400396.</text>
+              <element-id type="integer">1</element-id>
+              <element-type>Pathway</element-type>
+            </reference>
+          </references>
         </pathway>
         <cell-types/>
@@ -123,10 +131,10 @@
             <description nil="true"/>
             <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <kegg-id>C00885</kegg-id>
             <pubchem-cid nil="true"/>
-            <chebi-id nil="true"/>
+            <chebi-id>29780</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB02793</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -178,13 +186,13 @@
             <id type="integer">40792</id>
             <name>2,3-dihydroxybenzoic acid</name>
-            <hmdb-id nil="true"/>
+            <hmdb-id>HMDB0000397</hmdb-id>
             <description nil="true"/>
-            <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <cas>303-38-8</cas>
+            <kegg-id>C00196</kegg-id>
             <pubchem-cid nil="true"/>
             <chebi-id>18026</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB01672</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -206,11 +214,11 @@
           <compound>
             <id type="integer">108870</id>
-            <name>pyruvate</name>
+            <name>Pyruvic acid</name>
             <hmdb-id nil="true"/>
             <description nil="true"/>
-            <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <cas>57-60-3</cas>
+            <kegg-id>C00022</kegg-id>
             <pubchem-cid nil="true"/>
-            <chebi-id nil="true"/>
+            <chebi-id>15361</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
@@ -273,5 +281,5 @@
             <biocyc-id>NAD</biocyc-id>
             <chemspider-id>5682</chemspider-id>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB01907</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -332,5 +340,5 @@
             <biocyc-id>PANTETHEINE-P</biocyc-id>
             <chemspider-id>962</chemspider-id>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB03912</drugbank-id>
```

## idx 10 (P1), ids written: 0, title: Recent advances in taste transduction and signaling.

```diff
--- flag-off
+++ flag-on
@@ -3,5 +3,5 @@
   <named-for-id type="integer">1</named-for-id>
   <named-for-type>Pathway</named-for-type>
-  <cached-name>Generated Pathway</cached-name>
+  <cached-name>Recent advances in taste transduction and signaling.</cached-name>
   <cached-description></cached-description>
   <cached-subject>Metabolic</cached-subject>
@@ -18,10 +18,18 @@
         <pathway>
           <id type="integer">1</id>
-          <name>Generated Pathway</name>
+          <name>Recent advances in taste transduction and signaling.</name>
           <description></description>
           <subject>Metabolic</subject>
           <species-id type="integer">12</species-id>
           <sub-pathways/>
-          <references/>
+          <references>
+            <reference>
+              <id type="integer">1</id>
+              <pubmed-id>32185015</pubmed-id>
+              <text>Kinnamon SC, Finger TE: Recent advances in taste transduction and signaling. 2019. doi: 10.12688/f1000research.21099.1. PMID: 32185015.</text>
+              <element-id type="integer">1</element-id>
+              <element-type>Pathway</element-type>
+            </reference>
+          </references>
         </pathway>
         <cell-types>
@@ -434,106 +442,5 @@
         </transports>
         <interactions/>
-        <bound-visualizations>
-          <bound-visualization>
-            <id type="integer">39</id>
-            <biological-state-id type="integer">12</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">0</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">40</id>
-            <biological-state-id type="integer">13</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">140</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">41</id>
-            <biological-state-id type="integer">14</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">280</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">42</id>
-            <biological-state-id type="integer">15</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">420</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">43</id>
-            <biological-state-id type="integer">16</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">560</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">44</id>
-            <biological-state-id type="integer">17</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">700</y>
-            <width>3200</width>
```

## idx 3 (P0), ids written: 2, title: Recent advances in taste transduction and signaling.

```diff
--- flag-off
+++ flag-on
@@ -3,5 +3,5 @@
   <named-for-id type="integer">1</named-for-id>
   <named-for-type>Pathway</named-for-type>
-  <cached-name>Generated Pathway</cached-name>
+  <cached-name>Recent advances in taste transduction and signaling.</cached-name>
   <cached-description></cached-description>
   <cached-subject>Metabolic</cached-subject>
@@ -18,10 +18,18 @@
         <pathway>
           <id type="integer">1</id>
-          <name>Generated Pathway</name>
+          <name>Recent advances in taste transduction and signaling.</name>
           <description></description>
           <subject>Metabolic</subject>
           <species-id type="integer">1</species-id>
           <sub-pathways/>
-          <references/>
+          <references>
+            <reference>
+              <id type="integer">1</id>
+              <pubmed-id>32185015</pubmed-id>
+              <text>Kinnamon SC, Finger TE: Recent advances in taste transduction and signaling. 2019. doi: 10.12688/f1000research.21099.1. PMID: 32185015.</text>
+              <element-id type="integer">1</element-id>
+              <element-type>Pathway</element-type>
+            </reference>
+          </references>
         </pathway>
         <cell-types>
@@ -187,11 +195,11 @@
           <compound>
             <id type="integer">40034</id>
-            <name>H+</name>
+            <name>Hydrogen Ion</name>
             <hmdb-id nil="true"/>
             <description nil="true"/>
             <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <kegg-id>C00080</kegg-id>
             <pubchem-cid nil="true"/>
-            <chebi-id nil="true"/>
+            <chebi-id>15378</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
@@ -213,4 +221,5 @@
             <synonyms/>
             <element-states/>
+            <short-name>H+</short-name>
           </compound>
           <compound>
@@ -384,106 +393,5 @@
         </transports>
         <interactions/>
-        <bound-visualizations>
-          <bound-visualization>
-            <id type="integer">34</id>
-            <biological-state-id type="integer">12</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">0</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">35</id>
-            <biological-state-id type="integer">13</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">140</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">36</id>
-            <biological-state-id type="integer">14</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">280</y>
-            <width>3200</width>
-            <height>140</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">37</id>
-            <biological-state-id type="integer">15</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">420</y>
```

## idx 4 (P0), ids written: 12, title: The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB.

```diff
--- flag-off
+++ flag-on
@@ -3,5 +3,5 @@
   <named-for-id type="integer">1</named-for-id>
   <named-for-type>Pathway</named-for-type>
-  <cached-name>Generated Pathway</cached-name>
+  <cached-name>The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB.</cached-name>
   <cached-description></cached-description>
   <cached-subject>Metabolic</cached-subject>
@@ -18,10 +18,18 @@
         <pathway>
           <id type="integer">1</id>
-          <name>Generated Pathway</name>
+          <name>The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB.</name>
           <description></description>
           <subject>Metabolic</subject>
           <species-id type="integer">3</species-id>
           <sub-pathways/>
-          <references/>
+          <references>
+            <reference>
+              <id type="integer">1</id>
+              <pubmed-id>40400396</pubmed-id>
+              <text>Bin X, Pawelek PD: The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of the Escherichia coli isochorismatase EntB. 2025. doi: 10.1002/pro.70160. PMID: 40400396.</text>
+              <element-id type="integer">1</element-id>
+              <element-type>Pathway</element-type>
+            </reference>
+          </references>
         </pathway>
         <cell-types/>
@@ -118,10 +126,10 @@
             <description nil="true"/>
             <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <kegg-id>C00885</kegg-id>
             <pubchem-cid nil="true"/>
-            <chebi-id nil="true"/>
+            <chebi-id>29780</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB02793</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -144,13 +152,13 @@
             <id type="integer">40792</id>
             <name>2,3-dihydroxybenzoic acid</name>
-            <hmdb-id nil="true"/>
+            <hmdb-id>HMDB0000397</hmdb-id>
             <description nil="true"/>
-            <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <cas>303-38-8</cas>
+            <kegg-id>C00196</kegg-id>
             <pubchem-cid nil="true"/>
             <chebi-id>18026</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB01672</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -201,11 +209,11 @@
           <compound>
             <id type="integer">108870</id>
-            <name>pyruvate</name>
+            <name>Pyruvic acid</name>
             <hmdb-id nil="true"/>
             <description nil="true"/>
-            <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <cas>57-60-3</cas>
+            <kegg-id>C00022</kegg-id>
             <pubchem-cid nil="true"/>
-            <chebi-id nil="true"/>
+            <chebi-id>15361</chebi-id>
             <biocyc-id nil="true"/>
             <chemspider-id nil="true"/>
@@ -233,6 +241,6 @@
             <hmdb-id nil="true"/>
             <description nil="true"/>
-            <cas nil="true"/>
-            <kegg-id nil="true"/>
+            <cas>28384-96-5</cas>
+            <kegg-id>C05821</kegg-id>
             <pubchem-cid nil="true"/>
             <chebi-id>28855</chebi-id>
@@ -598,5 +606,5 @@
           <reaction>
             <id type="integer">50000</id>
```

## idx 6 (P1), ids written: 3, title: Canonical and noncanonical Wnt signaling: Multilayered mediators, signaling mechanisms and major signaling crosstalk.

```diff
--- flag-off
+++ flag-on
@@ -3,5 +3,5 @@
   <named-for-id type="integer">1</named-for-id>
   <named-for-type>Pathway</named-for-type>
-  <cached-name>Generated Pathway</cached-name>
+  <cached-name>Canonical and noncanonical Wnt signaling: Multilayered mediators, signaling mechanisms and major signaling crosstalk.</cached-name>
   <cached-description></cached-description>
   <cached-subject>Metabolic</cached-subject>
@@ -18,10 +18,18 @@
         <pathway>
           <id type="integer">1</id>
-          <name>Generated Pathway</name>
+          <name>Canonical and noncanonical Wnt signaling: Multilayered mediators, signaling mechanisms and major signaling crosstalk.</name>
           <description></description>
           <subject>Metabolic</subject>
           <species-id type="integer">1</species-id>
           <sub-pathways/>
-          <references/>
+          <references>
+            <reference>
+              <id type="integer">1</id>
+              <pubmed-id>37588235</pubmed-id>
+              <text>Qin K, Yu M, Fan J, Wang H, Zhao P, Zhao G, Zeng W, Chen C, Wang Y, Wang A, Schwartz Z, Hong J, Song L, Wagstaff W, Haydon RC, Luu HH, Ho SH, Strelzow J, Reid RR, He TC, Shi LL: Canonical and noncanonical Wnt signaling: Multilayered mediators, signaling mechanisms and major signaling crosstalk. 2024. doi: 10.1016/j.gendis.2023.01.030. PMID: 37588235.</text>
+              <element-id type="integer">1</element-id>
+              <element-type>Pathway</element-type>
+            </reference>
+          </references>
         </pathway>
         <cell-types/>
@@ -139,5 +147,5 @@
             <biocyc-id>GTP</biocyc-id>
             <chemspider-id>6569</chemspider-id>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB04137</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -168,5 +176,5 @@
             <biocyc-id>GDP-4-DEHYDRO-6-DEOXY-D-MANNOSE</biocyc-id>
             <chemspider-id>8630</chemspider-id>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB04315</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -227,5 +235,5 @@
             <biocyc-id>ADP</biocyc-id>
             <chemspider-id>5800</chemspider-id>
-            <drugbank-id nil="true"/>
+            <drugbank-id>DB03431</drugbank-id>
             <moldb-smiles/>
             <moldb-formula/>
@@ -624,5 +632,5 @@
           <reaction>
             <id type="integer">50000</id>
-            <spontaneous type="boolean">false</spontaneous>
+            <spontaneous nil="true" type="boolean"/>
             <pwr-id>PW_R050000</pwr-id>
             <direction>Right</direction>
@@ -678,5 +686,5 @@
           <reaction>
             <id type="integer">50001</id>
-            <spontaneous type="boolean">false</spontaneous>
+            <spontaneous nil="true" type="boolean"/>
             <pwr-id>PW_R050001</pwr-id>
             <direction>Right</direction>
@@ -714,5 +722,5 @@
           <reaction>
             <id type="integer">50002</id>
-            <spontaneous type="boolean">false</spontaneous>
+            <spontaneous nil="true" type="boolean"/>
             <pwr-id>PW_R050002</pwr-id>
             <direction>Right</direction>
@@ -810,76 +818,5 @@
           </interaction>
         </interactions>
-        <bound-visualizations>
-          <bound-visualization>
-            <id type="integer">49</id>
-            <biological-state-id type="integer">7</biological-state-id>
-            <x type="integer">0</x>
-            <y type="integer">0</y>
-            <width>3200</width>
-            <height>760</height>
-            <zindex type="integer">1</zindex>
-            <hidden type="boolean">false</hidden>
-          </bound-visualization>
-          <bound-visualization>
-            <id type="integer">50</id>
-            <biological-state-id type="integer">8</biological-state-id>
-            <x type="integer">0</x>
```

