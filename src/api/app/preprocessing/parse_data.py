from typing import Tuple

from regex import regex


def roman_to_int(roman_numeral: str) -> int:
    roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    int_val = 0
    for i in range(len(roman_numeral)):
        if (
            i > 0
            and roman_numerals[roman_numeral[i]] > roman_numerals[roman_numeral[i - 1]]
        ):
            int_val += (
                roman_numerals[roman_numeral[i]]
                - 2 * roman_numerals[roman_numeral[i - 1]]
            )
        else:
            int_val += roman_numerals[roman_numeral[i]]
    return int_val


def split_book(text: str) -> Tuple[str, str]:
    excerpt, book = regex.split(
        r"\s*\*\*\* START OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET \*\*\*\s*",
        text,
    )
    book, post_story_notice = regex.split(
        r"\s\*\*\* END OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET *\*\*\s*", text
    )
    epilogue_descriptions, book_text = regex.split(
        r"Scene III. A churchyard; in it a Monument belonging to the Capulets.", book
    )
    return (book, epilogue_descriptions)
