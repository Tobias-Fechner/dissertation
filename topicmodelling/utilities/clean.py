"""
Module used to pre-process raw text-like data. Can apply to strings containing HTML tags, for example. This module should contain
all functionality necessary to take in any arbitrary text-like data, and reduce it to clean, normalised text ready for
use to train topic models.

Does not include tokenization. Therefore outputs should be lowercase strings of words, not lists of tokens.
"""

import pandas as pd
from bs4 import BeautifulSoup
import unicodedata
import logging
import re
import pint
import string
import spacy
nlp = spacy.load("en_core_web_sm")

REGEX_PATTERNS = {
    'compoundedWords': re.compile(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))'),
    'specialCharacters': re.compile(f'[{string.punctuation}]'),
    'units': re.compile(' | '.join(list(pint.UnitRegistry())))
}

def dropHTML(html):
    soup = BeautifulSoup(html, features='lxml')
    raw = soup.get_text()
    t = unicodedata.normalize('NFKD', raw)
    return t

def dropDigits(text):
    logging.info(f"Dropped digits: {len([d for d in text if d.isdigit()])}.")
    return ''.join([i for i in text if not i.isdigit()])

def splitCompoundedWords(text):
    pattern = REGEX_PATTERNS['compoundedWords']
    logging.info(f"Split compounded words: {len(pattern.findall(text))}.")
    return pattern.sub(r'\1 ', text)

def dropSpecialChars(text):
    pattern = REGEX_PATTERNS['specialCharacters']
    logging.info(f"Dropped special characters: {len(pattern.findall(text))}. (Does not include dash, which is handled separately.)")
    t = pattern.sub('', text)
    return re.sub(' – ', ' ', t)

def dropUnits(text):
    pattern = REGEX_PATTERNS['units']
    logging.info(f"Dropped units: {pattern.findall(text)}.")
    return pattern.sub(' ', text)

def cleanHTML(htmlContent):
    # TODO: Log document id in each of these functions. Or record as extra columns the number of dropped items for each.
    assert isinstance(htmlContent, pd.Series)

    # Get raw text, removing html tags
    print("Dropping html tags.")
    raw = htmlContent.apply(lambda w: dropHTML(w))

    # Drop numeric characters and lowercase everything
    print("Dropping digits.")
    alpha = raw.apply(lambda x: dropDigits(x))

    # Drop all punctuation and special characters
    print("Dropping special characters.")
    noSpecials = alpha.apply(lambda y: dropSpecialChars(y))

    # Split compound words by capital letter and lower all text
    print("Splitting compounded words.")
    split = noSpecials.apply(lambda z: splitCompoundedWords(z))

    # Drop all units
    print("Dropping units.")
    noUnits = split.apply(lambda a: dropUnits(a))

    # Lowercase everything
    print("Making lower case.")
    lower = noUnits.apply(lambda b: b.lower())

    print("(Not) Checking cleanliness.") #TODO: Make sure can pass cleanliness test
    # assert all(lower.apply(lambda c: isHygienic(c))), "Text was cleaned but still fails cleanliness test."
    return lower

def isHygienic(text):
    """
    Function checks input text for all data cleaning processes defined so far and returns boolean indicating if text is clean for each case or not.
    :param text: input string
    :return: boolean indicating cleanliness
    """
    try:
        assert all(len(pattern.findall(text)) == 0 for pattern in REGEX_PATTERNS.values())
    except AssertionError:
        failing = [key for key in REGEX_PATTERNS.keys() if len(REGEX_PATTERNS[key].findall(text)) != 0]
        logging.error(f"Failing cleanliness check for {failing}.")
        for key in failing:
            print(f"Failed cleanliness check for {key} because the following characters were found: {REGEX_PATTERNS[key].findall(text)}")
        return False
    return True



