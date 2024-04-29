from typing import Any, Dict, List, Tuple

import pandas as pd
import regex
import tqdm
from preprocessing.ner import extract_people
from preprocessing.parse_data import roman_to_int, split_book
from transformers import pipeline

from preprocessing.api_logging import logger

def split_into_chapters(text: str) -> Dict[str, str | int]:
    # Split text into acts
    acts = regex.split(r"(?=ACT\s+[IVXLCDM]+)", text)
    acts = [act for act in acts if act.strip().startswith("ACT")]

    act_scene_data = []

    # Regex to capture scenes more reliably
    scene_regex = r"\bSCENE\s+[IVXLCDM]+\b"

    for act in acts:
        act_number_match = regex.search(r"ACT\s+([IVXLCDM]+)", act)
        if act_number_match:
            current_act_number = roman_to_int(act_number_match.group(1))
            scenes = regex.split(scene_regex, act[act_number_match.end() :])
            scenes = [scene.strip() for scene in scenes if scene.strip()]

            # Find scene numbers by re-applying the scene regex to capture the numbers
            scene_numbers = regex.findall(scene_regex, act)
            scene_numbers = [roman_to_int(num.split()[-1]) for num in scene_numbers]

            for scene_number, scene_text in zip(scene_numbers, scenes):
                act_scene_data.append(
                    {
                        "act": current_act_number,
                        "scene": scene_number,
                        "text": scene_text,
                        "number_words_in_scene": len(regex.split(r"[\s,\-;]+", text)),
                    }
                )

    return act_scene_data


def extract_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """ """
    # Define the pattern to split by
    pattern = r"\[\_.*?\_\]"

    # Function to apply to each row
    def split_into_blocks(row) -> List[str]:
        # Find all segments split by the pattern
        segments = regex.split(pattern, row["text"])
        segments = [seg.strip() for seg in segments if seg.strip()]

        # Find all instances of the pattern
        dividers = regex.findall(pattern, row["text"])

        # Combine dividers with text segments
        combined = []
        for seg, div in zip(
            segments, dividers + [""]
        ):  # Add empty string to handle last segment without a following divider
            combined.append(f"{seg} {div}".strip())

        return combined

    # Create a new column 'blocks' in the DataFrame by applying the function
    df["chunk_text"] = df.apply(split_into_blocks, axis=1)
    # Explode the DataFrame on the 'blocks' column to separate each block into its own row
    exploded_df = df.explode("chunk_text").reset_index(drop=True)

    return exploded_df


def make_chunks(
    text: str, find_people: bool = True
) -> List[Dict[str, str | int]] | None:
    """ """
    tqdm.tqdm.pandas()
    ner_pipeline = None
    if find_people:
        ner_pipeline = pipeline(
            "ner",
            model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
            grouped_entities=True,
        )

    logger.info("Preparing and chunking book")
    # Remove the filler sections from the book and split text from intro.
    book, intro_descriptions = split_book(text)
    # Split the book into a set of chapters and acts. Then split into chunks on exuent, died ect...
    chapters_dictionaries = split_into_chapters(book)
    chapters_df = pd.DataFrame(chapters_dictionaries)
    chunked_text_df: pd.DataFrame = extract_blocks(chapters_df)

    # Find the lengths of the acts and append to our dataframe
    act_text_length = (
        chunked_text_df.groupby("act")["number_words_in_scene"]
        .sum()
        .reset_index(name="num_words_in_act")
    )
    chunked_text_df = chunked_text_df.merge(act_text_length, on="act")

    # Find entities in the texts to help with metadata embedding
    logger.info("Extracting entities from passages to help with retrieval")
    chunked_text_df["possible_entities"] = chunked_text_df["text"].progress_apply(
        lambda text: extract_people(text, ner_pipeline)
    )
    # Reorder rows for downstream processing and remove redundant full text.
    chunked_text_df = chunked_text_df[
        [
            "act",
            "scene",
            "num_words_in_act",
            "number_words_in_scene",
            "possible_entities",
            "chunk_text",
        ]
    ]
    return [dict(v) for _, v in chunked_text_df.iterrows()]
