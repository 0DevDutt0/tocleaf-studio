# Corpora used for testing

Each corpus has a folder with its provenance and, where needed, a correction file. The correction file lists
outline entries to exclude and text lines to remove. Each rule has a reason, and the analysis records every
applied rule in its validation report. Rules are curation choices. Delete a rule to restore the entry.

| Corpus | Source | Licence | Committed here | Used for |
|---|---|---|---|---|
| `arxiv-2609.03874/` | STAIR paper, arXiv:2609.03874 (CC BY 4.0) | CC BY 4.0 | the PDF and provenance | simple PDF smoke test |
| `openstax-introduction-to-intellectual-property/` | OpenStax, *Introduction to Intellectual Property* | CC BY 4.0 | corrections and provenance only | real-document test |

## Files kept out of git

The OpenStax PDF is 24.6 MB, so it is not committed. Re-download it from the URL in
`openstax-introduction-to-intellectual-property/README.md`, and check the SHA-256 before use.

## Why the Nursing Assistant book is not used

The recommended CC BY 4.0 Nursing Assistant book (`wtcs.pressbooks.pub/nurseassist`) returned an error page to an
automated request on 10 Nov 2026. Its site does not permit automated downloads, so it was not retrieved, and it is not
used here.
