import pandas as pd
import nltk
from bs4 import BeautifulSoup
import unicodedata
import logging
import re
import pint

def tokenize(text):
    return nltk.word_tokenize(text)

def getWordsVocab(tokens):
    text = nltk.Text(tokens)
    words = [w.lower() for w in text]
    vocab = sorted(set(words))
    return words, vocab

def dropHTML(html):
    soup = BeautifulSoup(html, features='lxml')
    raw = soup.get_text()
    t = unicodedata.normalize('NFKD', raw)
    return t

def dropDigits(text):
    logging.info(f"Dropped digits: {len([d for d in text if d.isdigit()])}.")
    return ''.join([i for i in text if not i.isdigit()])

def splitCompoundedWords(text):
    pattern = re.compile(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))')
    logging.info(f"Split compounded words: {len(pattern.findall(text))}.")
    return pattern.sub(r'\1 ', text)

def dropSpecialChars(text):
    pattern = re.compile('[,.!?:%$£\[\]“”()]')
    logging.info(f"Dropped special characters: {len(pattern.findall(text))}. (Does not include dash, which is handled separately.)")
    t = pattern.sub('', text)
    return re.sub(' – ', ' ', t)

def dropUnits(text):
    uR = list(pint.UnitRegistry())
    pattern = re.compile(' | '.join(uR))
    logging.info(f"Dropped units: {pattern.findall(text)}.")
    return pattern.sub(' ', text)

def cleanHTML(col):
    # TODO: Log document id in each of these functions. Or record as extra columns the number of dropped items for each.
    assert isinstance(col, pd.Series)

    # Get raw text, removing html tags
    raw = col.apply(lambda w: dropHTML(w))

    # Drop numeric characters and lowercase everything
    alpha = raw.apply(lambda x: dropDigits(x))

    # Drop all punctuation and special characters
    noSpecials = alpha.apply(lambda y: dropSpecialChars(y))

    # Split compound words by capital letter and lower all text
    split = noSpecials.apply(lambda z: splitCompoundedWords(z))

    # Drop all units
    noUnits = split.apply(lambda a: dropUnits(a))

    # Lowercase everything
    lower = noUnits.apply(lambda b: b.lower())

    return lower



