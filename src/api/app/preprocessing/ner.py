from typing import Optional

import transformers


def extract_people(
    text: str,
    ner_model: Optional[
        transformers.pipelines.token_classification.TokenClassificationPipeline
    ] = None,
):
    if not ner_model:
        return []

    ner_results = ner_model(text)
    people = [
        entity["word"] for entity in ner_results if entity["entity_group"] == "PER"
    ]
    return list(set(people))
