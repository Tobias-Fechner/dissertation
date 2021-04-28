import topicmodelling.utilities.clean
import apiIntegrations.utilities

import pandas as pd
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

import pathlib, logging
import spacy, gensim

def getTokens(texts):
    assert isinstance(texts, pd.Series)
    tqdm.pandas()
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "ner"])
    logging.info("Getting tokens.")
    return texts.progress_apply(lambda x: nlp(x))

class Corpus:
    def __init__(self, name):
        assert isinstance(name, str), "Please set corpus name with string."

        self.name = name
        self.source = None
        self.sourceType = None
        self.raw = None
        self.data = None
        self.dictionary = None
        self.location = None

    def addData(self, source='ga', sourceType='api'):

        self.source = source
        self.sourceType = sourceType

        dataDir = pathlib.Path(pathlib.Path.cwd().parent, 'data')
        if self.source == 'ga' and self.sourceType == 'api':
            parentDir = pathlib.Path(dataDir, f'{sourceType}Integrations.{source}')
        else:
            raise NotImplementedError

        print("Available files: ")

        availableFiles = [(count, f"{count}: {str(filePath.stem)}") for count, filePath in enumerate(parentDir.iterdir())]
        print("\n".join(list(zip(*availableFiles))[1]))

        selection = '-1'

        while int(selection) not in list(zip(*availableFiles))[0]:
            print("Specify file to select by number (esc to break):")
            selection = input("--> ")
            if selection == 'esc':
                selection = ''
                break

        self.location = list(parentDir.iterdir())[int(selection)]
        assert self.location.exists(), "Filepath to data source does not exist."
        assert self.location.suffix == '.csv', "Must load data from csv."

        self.raw = pd.read_csv(self.location, index_col=0, usecols=apiIntegrations.utilities.getApiColsOfInterest(self.source))
        return

    def updateDictionary(self, tokensProcessed):
        self.dictionary = gensim.corpora.dictionary.Dictionary(tokensProcessed)

        # Filter out words that occur in less than 3 documents, or more than 60% of the documents.
        self.dictionary.filter_extremes(no_below=3, no_above=0.6)
        return

    def checkDataClean(self):

        textColName = 'text'

        # Check data is clean
        try:
            assert all(topicmodelling.utilities.clean.isHygienic(text) for text in self.raw[textColName])
        except KeyError:
            print(
                f"Required column '{textColName}' not found. Please identify the column containing clean text data: \n{self.raw.columns}")
            textColName = input("-->")
            assert all(topicmodelling.utilities.clean.isHygienic(text) for text in self.raw[textColName])
            logging.info(f"Column {textColName} identified by user as containing clean text data.")
        except AssertionError:
            logging.debug(
                "Failed to add data due to data cleanliness issues. Refer to module topicmodelling.utilities.clean.")
            raise AssertionError("Document text must be clean before storing in corpus.")

        return

    def prepareData(self, texts):
        assert isinstance(texts, pd.Series)

        tokens = getTokens(texts)
        tokensProcessed = topicmodelling.utilities.clean.cleanTokens(tokens)

        self.updateDictionary(tokensProcessed)
        dtm = tokensProcessed.apply(lambda x: self.dictionary.doc2bow(x))

        self.data = pd.DataFrame(index=texts.index, data={
            'texts': texts,
            'tokens': tokens,
            'tokensProcessed': tokensProcessed,
            'dtm': dtm
        })
        return

class TopicModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def getDominantTopics(self):
        pass

    @abstractmethod
    def getSingleDominantTopic(self):
        pass

    @abstractmethod
    def getDominantDocuments(self):
        pass

class LDA(TopicModel):
    def __init__(self):
        super().__init__()

    def getDominantTopics(self):
        pass

    def getSingleDominantTopic(self):
        pass

    def getDominantDocuments(self):
        pass

    raise NotImplementedError

class HDP(TopicModel):
    def __init__(self):
        super().__init__()

    def getDominantTopics(self):
        pass

    def getSingleDominantTopic(self):
        pass

    def getDominantDocuments(self):
        pass

    raise NotImplementedError




