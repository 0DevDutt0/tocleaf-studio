# OpenStax, Introduction to Intellectual Property (real-document test)

- Source: OpenStax asset host, https://assets.openstax.org/oscms-prodcms/media/documents/introduction-intellectual-property_-_WEB.pdf
  (linked from openstax.org/details/books/introduction-intellectual-property)
- Licence: CC BY 4.0. Attribution: "OpenStax, Introduction to Intellectual Property, CC BY 4.0".
- File: 24,582,768 bytes, 201 pages, SHA-256 `1498509864d37cb63a6ccc11bab7c783f3d30b2ed37d808ef80c56fdb676378a`.
  The PDF itself is not committed (24.6 MB). Download it from the URL above and check the SHA-256.
- Native outline before corrections: 83 ToC nodes (75 leaves). After `corrections.json`: 52 nodes, 47 leaves, validation passes.
- Corrections: 20 excluded outline entries (front and back matter, chapter-opening entries that conflict with the printed
  numbering, assessment question sets, and the glossary and answer key), plus 2 line rules that remove running heads.
  Every rule has a reason, and the validation report records how many words each one removed.
- Cross-check: the SearchTome release (CC BY-SA 4.0) has a law corpus of 47 documents from this book. Those are section-level
  documents with titles of the form "1.1 ...", and the counts match the leaves above.

## Author-written query set (frozen 10 Nov 2026)

- File: `author_queries.jsonl`, 147 rows. 141 answerable queries: three per leaf for all 47 leaves (`direct`,
  `paraphrase`, `contextual`). 6 out-of-scope queries (`gold_section: null`), reported separately.
- SHA-256 `060c93c45d7137d3deeb30acd93c61fbb61c627ab2ed9f29e4bd58b3e09b0ea8`. Frozen before any model or BM25 run used it.
  Make no edits to the file now; a changed set needs a new name and a new freeze record.
- Written by the project's engineering agent, not an independent annotator. The writer read each leaf's opening text
  (the first ~260 words of `docs.jsonl`, plus `toc.json`). It did not read the synthetic train, validation or test
  queries, and it had seen no results on this set. This is the first query set in the project that no configuration
  was tuned on.
- Leakage screen after writing: seven queries had a content-token Jaccard of at least 0.6 with a synthetic question. They
  were rewritten, and the originals and reasons are in `docs/results/openstax-ip-author/screening_log.json`. After the
  rewrite, no query reaches 0.6 against any of the 744 synthetic questions.
- Lexical overlap with the gold leaf (mean share of query content words found in the leaf): 0.692 for this set (direct
  0.842, paraphrase 0.670, contextual 0.564), against 0.839 for the synthetic test split. This set is less lexically
  aligned with its passages, as intended.
- Gold labels are leaf section numbers. The loader rejects a parent node.
