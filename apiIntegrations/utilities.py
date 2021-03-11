import logging
import pandas as pd
import pathlib
import os
from bs4 import BeautifulSoup
import unicodedata
import re
import pint

COMPARISON_SKIP_COLS = {
    'channels': [],
    'channelsSearch': [0,-2,-1],
    'summaries': [],
}

def depositData(data, parentFolder, fileName, fileType='csv'):

    dataDir = pathlib.Path(os.getcwd(), 'data', parentFolder)

    confirm = 'a'
    while confirm not in ['y', 'Y', 'n', 'N']:
        print(f"Please confirm you want to deposit data to the following folder. Note, data will be discarded if not. \n{dataDir}")
        confirm = input("(y/n) -->")

    if confirm == 'n' or confirm == 'N':
        logging.info("Declined data folder location. Data discarded.")
        print("New data discarded.")

    else:
        if not dataDir.exists():
            os.mkdir(dataDir)
        else:
            pass

        filePath = pathlib.Path(dataDir, f"{fileName}.{fileType}")

        if not filePath.exists():
            print(f"Please confirm the following file path is correct. Note, data will be discarded if not. \n{filePath}")
            confirm = 'a'
            while confirm not in ['y', 'Y', 'n', 'N']:
                print(
                    f"Please confirm you want to deposit data to the following folder. Note, data will be discarded if not. \n{dataDir}")
                confirm = input("(y/n) -->")
            if confirm == 'n' or confirm == 'N':
                logging.info("Declined data folder location. Data discarded.")
                print("New data discarded.")
                pass
            else:
                data.to_csv(filePath)

        else:
            if not filePath.stem in COMPARISON_SKIP_COLS.keys():
                print(f"Data deposit method not yet defined for this request type. Need a new key '{filePath.stem}' in the COMPARISON_SKIP_COLS dictionary.")
                tryOverwriteFile(data, filePath)

            dataOnDisk = pd.read_csv(filePath, index_col=0)

            # Get columns that should be skipped. This is because dictionaries are written as strings to csv.
            skipCols = COMPARISON_SKIP_COLS[filePath.stem]

            # Drop the troublesome columns from each dataframe
            df1 = data.drop(data.columns[skipCols], axis=1)
            df2 = dataOnDisk.drop(dataOnDisk.columns[skipCols], axis=1)

            # If data is the same, overwrite straight away. Otherwise, ask for user confirmation.
            if df1.equals(df2):
                logging.info("Detected no change in data, so overwrote data on disk.")
                data.to_csv(filePath)

            else:
                logging.warning("Detected change in data.")
                tryOverwriteFile(data, filePath)

def tryOverwriteFile(data, filePath):

    confirmation = 'Pancakes taste nice on a Sunday morning.'

    while confirmation not in ['n', 'N', 'y', 'Y']:
        print("Detected change in data. Are you sure you want to overwrite? (y/n)")
        confirmation = input("-->")

    if confirmation in ['y', 'Y']:
        logging.info("Data on disk overwritten with new data.")
        data.to_csv(filePath)

    else:
        logging.info("New data discarded. Kept data on disk.")
        print("Data discarded.")

    return

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

def getPlainTextFromHTML(col):
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

    # Drop all punctuation, special characters, and units
    noUnits = split.apply(lambda a: dropUnits(a))

    # Lowercase everything
    lower = noUnits.apply(lambda b: b.lower())

    return lower

