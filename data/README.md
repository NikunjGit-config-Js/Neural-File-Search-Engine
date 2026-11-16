# Data

## `sample_documents/` — synthetic demonstration corpus

These 32 files are **synthetic**. They were written for this project and describe a fictional
parcel-delivery company called "Brightfield Logistics". They are not real company documents.
Each file also records this in its metadata (PDF subject, DOCX comments, PPTX subject).

| Format | Files | Unit used as a section |
|---|---|---|
| PDF  | 8 policies and reports | page |
| DOCX | 8 procedures, guides and minutes | heading section |
| PPTX | 8 slide decks | slide |
| CSV  | 8 tables (tickets, catalogue, inventory, ...) | row |

Regenerate them with `python prepare_data.py`. The script is deterministic.

## `queries.csv`

44 queries: `query_id`, `query_type`, `query`.

| query_type | count | purpose |
|---|---|---|
| keyword | 11 | wording copied from the documents; TF-IDF should do well |
| paraphrase | 20 | natural questions using different words from the documents |
| ambiguous | 8 | one- or two-word queries with several relevant files |
| no_answer | 5 | nothing in the corpus answers them |

## `relevance_labels.csv`

`query_id`, `file_name`, `relevance` (always 1). Labels are **binary and at document level**.
Any file not listed for a query counts as not relevant, and `no_answer` queries have no rows.
The labelling rule was: a file is relevant if someone asking the query would want to open it
because it substantially answers the question; a passing mention is not enough.

The labels were written together with the queries, before any model was run. They were not
changed after looking at model output. The same author wrote the documents, the queries and the
labels, which is a source of bias. See "Limitations" in the main README.
