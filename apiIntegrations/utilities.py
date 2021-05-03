import logging
import pandas as pd
import pathlib
import os

COMPARISON_SKIP_COLS = {
    'channels': [],
    'channelsSearch': [0,-2,-1],
    'summaries': [],
}

def depositData(data, parentFolder, fileName, fileType='csv'):

    requiredCols = ['title','author','text']

    assert isinstance(data, pd.DataFrame)
    assert all(colName in data.columns for colName in requiredCols), f"Data requires columns {requiredCols}. Columns present: {data.columns}."

    dataDir = pathlib.Path(os.getcwd(), 'data', parentFolder)

    confirm = 'a'
    while confirm not in ['y', 'Y', 'n', 'N']:
        print(f"Please confirm you want to deposit data to the following folder. Note, data will be discarded if not. \n{dataDir}")
        confirm = input("(y/n) -->")

    if confirm in ['n','N']:
        logging.info("Declined data folder location. Data discarded.")
        print("New data discarded.")
        return

    if not dataDir.exists():
        os.mkdir(dataDir)

    filePath = pathlib.Path(dataDir, f"{fileName}.{fileType}")

    if filePath.exists():
        tryOverwriteFile(data, filePath)
    else:
        confirm = 'a'
        while confirm not in ['y', 'Y', 'n', 'N']:
            print(f"Please confirm the following file path is correct: {filePath} \nNote, data will be discarded if not. (y/n)")
            confirm = input("-->")

        if confirm == 'n' or confirm == 'N':
            logging.info("File path was incorrect. Data discarded.")
            print("New data discarded.")
            pass
        else:
            data.to_csv(filePath)

def tryOverwriteFile(data, filePath):

    confirmation = 'Pancakes taste nice on a Sunday morning.'

    while confirmation not in ['n', 'N', 'y', 'Y']:
        print(f"Detected existing file: {filePath} \nAre you sure you want to overwrite? (y/n)")
        confirmation = input("-->")

    if confirmation in ['y', 'Y']:
        logging.info("Data on disk overwritten with new data.")
        data.to_csv(filePath)

    else:
        logging.info("New data discarded. Kept data on disk.")
        print("Data discarded.")

    return

def getApiColsOfInterest(api):
    assert isinstance(api, str), "Must pass api name as string."
    store = {
        'ga': ['id', 'title', 'subtitle', 'author',  'publisher', 'introHtml', 'contentHtml', 'source', 'language'],
    }

    assert api in store.keys()
    return store[api]


