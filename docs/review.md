# Interactive Review & ATLAS Export

After running Phecoder, you can interactively curate the top-K retrieved ICD codes per phecode in a Jupyter notebook, then export the results for use in cohort tools like OHDSI ATLAS.

## Install the review extra

```bash
pip install 'phecoder[review]'
```

## Review results

```python
session = pc.review(top_k=30, score_threshold=0.5)
session  # renders a tabbed checkbox widget, one tab per phecode
```

After curating in the widget (or programmatically), save your selections:

```python
session.save("picks.parquet")   # flat table with provenance
session.save("picks.json")      # nested concept-set JSON, grouped by phecode
session.save("picks.csv")       # flat CSV
```

## Export to OHDSI ATLAS

ATLAS concept sets are defined in terms of OMOP `concept_id`s, not raw ICD codes. To use the ATLAS exporter, add a `concept_id` column (and optionally `vocabulary_id`) to your `icd_df` **before** instantiating `Phecoder` — any extra columns on `icd_df` are preserved end-to-end and become available to the exporter:

```python
from phecoder import Phecoder, load_icd_df

icd_df = load_icd_df()
icd_df = icd_df.merge(my_omop_concept_map, on="icd_code", how="left")
#                    ^ supplies concept_id (+ vocabulary_id)

pc = Phecoder(phecodes=..., icd_df=icd_df, output_dir="./out")
pc.run()
pc.build_ensemble()

session = pc.review(top_k=30)
session  # curate, then:

pc.export_atlas(session, "./atlas_concept_sets")       # one JSON per phecode
pc.export_atlas(session, "./atlas_concept_sets.json")  # single bundle file
```

Import the resulting JSON via the ATLAS UI (**Concept Sets → Import**). The exporter sets `includeDescendants=true` and `includeMapped=true` by default, so ATLAS will follow the vocabulary hierarchy and ICD→SNOMED `Maps to` relationships at query time.

!!! tip "Where to get OMOP concept IDs"
    OMOP `concept_id`s for ICD vocabularies can be obtained from the OHDSI [Athena](https://athena.ohdsi.org) vocabulary download. Select `ICD9CM`, `ICD10`, `ICD10CM`, and `SNOMED`.
