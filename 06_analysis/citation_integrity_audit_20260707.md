# Citation Integrity Audit

Date: 2026-07-07

Scope:

- `07_paper/ieee_twc_isac_nd/main.tex`
- `07_paper/ieee_twc_isac_nd/references.bib`

## Summary

| Check | Result |
|---|---:|
| Unique in-text citation keys | 25 |
| Bibliography entries | 25 |
| In-text keys missing from `.bib` | 0 |
| `.bib` entries not cited in `main.tex` | 0 |
| DOI entries verified through Crossref DOI API | 22 |
| Entries without DOI after audit | 3 |

## Corrections Applied

| Key | Issue | Correction |
|---|---|---|
| `Vasudevan2005DirectionalND` | DOI missing | Added `10.1109/INFCOM.2005.1498535`; Crossref title matches "On neighbor discovery in wireless networks with directional antennas". |
| `Fiedler1973AlgebraicConnectivity` | DOI missing | Added `10.21136/CMJ.1973.101168`; Crossref title matches "Algebraic connectivity of graphs". |
| `Eisen2020REGNN` | DOI missing | Added `10.1109/TSP.2020.2988255`; Crossref title matches "Optimal Wireless Resource Allocation With Random Edge Graph Neural Networks". |
| `ThreeGPP22137ISAC` | Version-year mismatch | Changed year from 2026 to 2024 for TS 22.137 v19.1.0 and added the official 3GPP DynaReport URL. |
| `ThreeGPP38901Channel` | Missing official URL | Added the official 3GPP DynaReport URL for TR 38.901. |

## Remaining No-DOI Entries

| Key | Reason | Verification path |
|---|---|---|
| `ThreeGPP22137ISAC` | 3GPP technical specification; DOI not expected. | Official 3GPP DynaReport page: `https://www.3gpp.org/dynareport/22137.htm`. |
| `ThreeGPP38901Channel` | 3GPP technical report; DOI not expected. | Official 3GPP DynaReport page: `https://www.3gpp.org/dynareport/38901.htm`. |
| `Skolnik2001Radar` | Textbook reference; no DOI retained. | Standard book metadata retained: M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001. |

## Audit Trail

Commands executed:

```powershell
# Cross-check in-text cite keys against .bib keys.
$tex = Get-Content -Raw '07_paper/ieee_twc_isac_nd/main.tex'
$bib = Get-Content -Raw '07_paper/ieee_twc_isac_nd/references.bib'
```

```powershell
# DOI verification used Crossref's public DOI endpoint:
https://api.crossref.org/works/<doi>
```

External authoritative pages used:

- Crossref DOI API for all DOI-bearing entries.
- 3GPP DynaReport for TS 22.137: `https://www.3gpp.org/dynareport/22137.htm`.
- 3GPP DynaReport for TR 38.901: `https://www.3gpp.org/dynareport/38901.htm`.

## Remaining Integrity Risks

1. This audit verifies citation-key consistency and bibliographic existence/metadata for DOI-bearing entries. It does not prove that every cited sentence is semantically faithful to the cited paper.
2. A final submission pass should still perform claim-to-source alignment on the most important related-work sentences.
3. Retraction screening was not performed in this pass.
