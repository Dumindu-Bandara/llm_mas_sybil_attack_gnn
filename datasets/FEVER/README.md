# Data Format

The data is distributed in [JSONL](http://jsonlines.org) format, with one example per line.

## Training/Development Data Format

The training and development data contain 4 fields:

- **id**: The ID of the claim.
- **label**: The annotated label for the claim. Can be one of `SUPPORTS` | `REFUTES` | `NOT ENOUGH INFO`.
- **claim**: The text of the claim.
- **evidence**: A list of evidence sets (lists of `[Annotation ID, Evidence ID, Wikipedia URL, sentence ID]` tuples), or a `[Annotation ID, Evidence ID, null, null]` tuple if the label is `NOT ENOUGH INFO`.

  (The Annotation ID and Evidence ID fields are for internal use only and are not used for scoring. They may help debug or correct annotation issues at a later point in time.)

Below are examples of the data structures for each of the three labels.

### Supports Example

```json
{
    "id": 62037,
    "label": "SUPPORTS",
    "claim": "Oliver Reed was a film actor.",
    "evidence": [
        [
            [<annotation_id>, <evidence_id>, "Oliver_Reed", 0]
        ],
        [
            [<annotation_id>, <evidence_id>, "Oliver_Reed", 3],
            [<annotation_id>, <evidence_id>, "Gladiator_-LRB-2000_film-RRB-", 0]
        ],
        [
            [<annotation_id>, <evidence_id>, "Oliver_Reed", 2],
            [<annotation_id>, <evidence_id>, "Castaway_-LRB-film-RRB-", 0]
        ],
        [
            [<annotation_id>, <evidence_id>, "Oliver_Reed", 1]
        ],
        [
            [<annotation_id>, <evidence_id>, "Oliver_Reed", 6]
        ]
    ]
}
```

### Refutes Example

```json
{
    "id": 78526,
    "label": "REFUTES",
    "claim": "Lorelai Gilmore's father is named Robert.",
    "evidence": [
        [
            [<annotation_id>, <evidence_id>, "Lorelai_Gilmore", 3]
        ]
    ]
}
```

### NotEnoughInfo Example

```json
{
    "id": 137637,
    "label": "NOT ENOUGH INFO",
    "claim": "Henri Christophe is recognized for building a palace in Milot.",
    "evidence": [
        [
            [<annotation_id>, <evidence_id>, null, null]
        ]
    ]
}
```

## Test Data Format

The test data follows the same format as the training/development examples, with the `label` and `evidence` fields removed.

```json
{
    "id": 78526,
    "claim": "Lorelai Gilmore's father is named Robert."
}
```
