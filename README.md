# Profilet e komunave të Kosovës

Projekt për përgatitjen dhe analizën e të dhënave demografike të komunave të Kosovës, bazuar në tabelën e ASK-së për popullsinë sipas vitit, komunës, llojit të vendbanimit, gjinisë dhe grupmoshës.

## Qëllimi dhe shtrirja

Kjo fazë përgatit të dhënat për analiza të mëtejshme. Ajo përfshin kontrollin e cilësisë, pastrimin, validimin, agregimin dhe krijimin e veçorive demografike. Përgatitja për modelim përfshin gjithashtu përzgjedhjen dhe standardizimin e veçorive dhe reduktimin e dimensionit me PCA.

Analizat e gjeneratave, grupimi i komunave (clustering) dhe dashboard-i interaktiv janë punë të fazave të ardhshme. Rezultatet e kësaj faze nuk duhen interpretuar si matje të drejtpërdrejta të emigrimit.

## Burimi i të dhënave

- **Publikuesi:** Agjencia e Statistikave të Kosovës (ASK)
- **Tabela:** `census2024_13`
- **Vitet:** 2011 dhe 2024
- **Ndarjet:** komuna, urban/rural, gjini dhe 18 grupmosha
- **Skedari burimor:** `data/raw/askdata.xlsx`
- **Informacioni i burimit dhe SHA-256:** `data/raw/source.json`
- **Faqja e ASK-së:** [Popullsia sipas llojit të vendbanimit, gjinisë dhe grupmoshës](https://askdata.rks-gov.net/pxweb/sq/ASKdata/ASKdata__Census%20population__1_Demographic_Characteristics/census2024_13.px/)

Skedari origjinal kontrollohet kundrejt hash-it të regjistruar para dhe pas ekzekutimit të workflow-së. Data origjinale e shkarkimit nuk është regjistruar.

## Përgatitja e mjedisit

Kërkohet Python dhe PowerShell në Windows. Nga direktoria kryesore e projektit, krijoni mjedisin virtual dhe instaloni varësitë:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Varësitë përcaktohen në `requirements.txt` (pandas, NumPy dhe openpyxl).

## Ekzekutimi

Për të ekzekutuar të gjithë workflow-n për 2024:

```powershell
.\.venv\Scripts\python.exe run_phase1.py --year 2024
```

Për 2011:

```powershell
.\.venv\Scripts\python.exe run_phase1.py --year 2011
```

Viti i parazgjedhur është 2024. Skripti ekzekuton hapat në rend, ndalon nëse një hap dështon dhe në fund nis testet automatike. Përmbledhja dhe log-u ruhen te `results/phase1_summary_<viti>.txt` dhe `results/phase1_run_<viti>.txt`.

## Hapat e përpunimit

| Skripti | Përgjegjësia |
|---|---|
| `phase1/01_data_quality.py` | Auditimi fillestar i fletës Excel: madhësia, vlerat e zbrazëta, tipet, përsëritjet dhe përmbledhja e të dhënave bruto. |
| `phase1/02_preprocessing.py` | Heqja e rreshtave të titujve/shënimeve nga të dhënat analitike, kontrolli i strukturës, kthimi në format të gjatë dhe përcaktimi i tipeve. |
| `phase1/validate_data.py` | Krahasimi i çdo qelize të përpunuar me burimin dhe kontrolli i çelësave, kategorive dhe paralajmërimeve të mbulimit. |
| `phase1/03_aggregation.py` | Agregimi sipas komunës, vitit, vendbanimit dhe gjinisë, me tregues për plotësinë e qelizave. |
| `phase1/04_feature_preparation.py` | Përgatitja e dataset-it analitik, krijimi i grupeve 0–14, 15–64 dhe 65+ dhe i treguesve Boolean. |
| `phase1/05_demographic_features.py` | Llogaritja e popullsisë sipas grupmoshave dhe e treguesve demografikë. |
| `phase1/06_feature_scaling.py` | Përzgjedhja e veçorive për komunë, evidentimi i përjashtimeve dhe standardizimi për vitin e zgjedhur. |
| `phase1/07_pca.py` | Reduktimi i dimensionit me PCA përmes SVD-së; parazgjedhja ruan të paktën 90% të variancës. |

### Trajtimi i kërkesës për mostrimin

Skripti `phase1/08_sampling.py` përdoret vetëm për demonstrim dhe verifikim të shpejtë të funksionimit të workflow-it, duke krijuar një mostër të vogël të komunave për paraqitje dhe kontroll të ekzekutimit.

Kjo mostër nuk përdoret për analizat demografike, përgatitjen e veçorive, ose për modelimin. Për këto procese përdoren të gjitha regjistrimet e vlefshme të dataset-it, me përjashtim të rasteve kur zbatohen rregullat e dokumentuara të cilësisë dhe mbulimit të të dhënave.

Nuk është kryer mostër statistikore për analizën finale. Dataset-i i disponueshëm përdoret në tërësi për të ruajtur mbulimin e komunave, viteve, gjinive, vendbanimeve dhe kategorive demografike. Rreshtat e shfaqur në terminal janë vetëm parapamje të të dhënave dhe nuk përfaqësojnë mostër analitike.


## Rregullat dhe kufizimet e të dhënave

- Numërimet ruhen si numra të plotë të anulueshëm (`Int64`); viti si numër i plotë; etiketat si tekst.
- Vlerat e zbrazëta dhe shenja `:` trajtohen si mungesa; nuk imputohen dhe nuk kthehen në zero.
- Sipas shënimeve të burimit, katër komunat veriore nuk janë regjistruar më 2011. Vlerat zero të eksportuara për to paraqiten si të panjohura (`missing`), jo si popullsi zero.
- Pjesëmarrja në komunat veriore më 2024 është e paplotë. Numrat ruhen, por shoqërohen me paralajmërim.
- Agregimet shënojnë veçmas nëse qelizat burimore janë të plota; kjo nuk provon se mbulimi i regjistrimit është i plotë.
- Përqindjet dhe raportet mbeten të papërcaktuara kur mungon një vlerë e nevojshme ose emëruesi është zero.
- Totali me `settlement_type=all` është totali i komunës. Nuk duhet mbledhur përsëri me rreshtat urban dhe rural.
- PCA dhe standardizimi përshtaten veçmas për secilin vit. Prandaj, koordinatat e PCA-së dhe vlerat e standardizuara të viteve 2011 dhe 2024 nuk janë në të njëjtën bazë dhe nuk duhen krahasuar drejtpërdrejt.
- Përgatitja aktuale për modelim përjashton komunat me paralajmërime specifike të mbulimit dhe ruan arsyet e përjashtimit. Daljet e tjera analitike i mbajnë paralajmërimet.
- Ndryshimi i popullsisë mes dy regjistrimeve nuk mat drejtpërdrejt emigrimin; ndikohet edhe nga faktorë të tjerë dhe kufizimet e mbulimit.

## Daljet

Daljet e përpunuara ruhen nën `data/processed/`:

- `population.csv`, `population_coverage.csv`, `population_validated.csv`
- `aggregates/`: tabela të agreguara në nivele të ndryshme
- `analysis/population_analysis_ready.csv`
- `analysis/age_band_population.csv` dhe `analysis/demographic_indicators.csv`
- `modeling/`: veçoritë e përzgjedhura e të standardizuara, parametrat e standardizimit, komunat e përjashtuara dhe rezultatet e PCA-së

Raportet e cilësisë, përpunimit, agregimit, veçorive, validimit, standardizimit dhe PCA-së ruhen te `results/`. Skedarët e gjeneruar nën `data/processed/` dhe `results/` përjashtohen nga Git; workflow-ja i rigjeneron.

## Testet

Workflow-ja kryen testet me `unittest`. Për t’i nisur veçmas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Testet mbulojnë formulat dhe mungesat në treguesit demografikë, përzgjedhjen/standardizimin e veçorive dhe sjelljen e PCA-së.