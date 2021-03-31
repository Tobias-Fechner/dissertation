import topicmodelling.utilities.clean
import apiIntegrations.utilities

import pandas as pd
import numpy as np
from tqdm import tqdm

import pathlib, string, logging

import nltk, spacy, gensim

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

        # Filter out words that occur in no less than 10 documents, or more than 50% of the documents.
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
        tokensProcessed = cleanTokens(tokens)

        self.updateDictionary(tokensProcessed)
        dtm = tokensProcessed.apply(lambda x: self.dictionary.doc2bow(x))

        self.data = pd.DataFrame(index=texts.index, data={
            'texts': texts,
            'tokens': tokens,
            'tokensProcessed': tokensProcessed,
            'dtm': dtm
        })
        return

def getTokens(texts):
    assert isinstance(texts, pd.Series)
    tqdm.pandas()
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "ner"])
    logging.info("Getting tokens.")
    return texts.progress_apply(lambda x: nlp(x))

def cleanTokens(tokens):
    assert isinstance(tokens, pd.Series)
    tqdm.pandas()

    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(['could', 'also', 'get', 'use', 'us', 'since', 'would', 'may', 'however', 'well', 'must',
                      'much', 'even', 'like', 'many', 'one', 'two', 'new', 'every', 'recommends',
                      'large', 'less', 'more', 'though', 'yet', 'make', 'three', 'getabstract'])

    logging.info("Getting useful words: dropping stopwords, punctuation, non-alpha and tokens with only one letter.")
    docs = tokens.progress_apply(lambda x: getUsefulWords(x, stopwords))

    logging.info("Getting bigrams.")
    bigramModel = gensim.models.Phrases(docs, min_count=5)
    bigrams = docs.progress_apply(lambda x: np.array([token for token in bigramModel[x] if '_' in token]))

    return docs.apply(list) + bigrams.apply(list)

def getUsefulWords(tokens, stopwords):
    lemmas = np.array([token.lemma_.lower() for token in tokens])
    maskStopwords = np.array([lemma not in stopwords for lemma in lemmas], dtype=bool)
    maskPunctuation = np.array([lemma not in string.punctuation for lemma in lemmas], dtype=bool)
    maskNumeric = np.array([lemma.isalpha() for lemma in lemmas], dtype=bool)
    maskLength = np.array([len(lemma) > 1 for lemma in lemmas], dtype=bool)
    return lemmas[maskStopwords & maskPunctuation & maskNumeric & maskLength]

def getNounChunks(docText, stopwords):
    chunksAsText = np.array([chunk.text for chunk in docText.noun_chunks if len(chunk) > 1])
    chunksAsSpan = [chunk for chunk in docText.noun_chunks if len(chunk) > 1]
    maskChunksStopwords = np.array([all(ch.text not in stopwords for ch in chunk) for chunk in chunksAsSpan],
                                   dtype=bool)
    maskChunksNumeric = np.array([all(ch.text.isalpha() for ch in chunk) for chunk in chunksAsSpan], dtype=bool)
    return chunksAsText[maskChunksStopwords & maskChunksNumeric]





