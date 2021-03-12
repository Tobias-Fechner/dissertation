import topicmodelling.utilities.clean
import apiIntegrations.utilities
import pandas as pd
import pathlib
import logging
import nltk
import itertools

class Document:
    def __init__(self, iD, text, language='en'):
        assert topicmodelling.utilities.clean.isHygienic(text), "Must pass clean text data. Refer to module topicmodelling.utilities.clean."

        self.id = iD
        self.text = text
        self.language = language

        raise NotImplementedError

    def getBagOfWords(self):
        """
        Take in text string, and create list of non-stopword words.
        :return: list of words
        """
        stopwords = nltk.corpus.stopwords.words('english')
        return sorted(set(word for word in self.text.split() if word not in stopwords))

class Corpus:
    def __init__(self, name):
        assert isinstance(name, str), "Please set corpus name with string."

        self.name = name
        self.source = None
        self.sourceType = None
        self.data = None
        self.vocabulary = None
        self.location = None

    def addData(self, source='ga', sourceType='api', filePath=r'C:\Users\Tobias Fechner\Documents\1_Uni\fyp\git_repo_fyp\data\apiIntegrations.ga\energy_environment_sustainableenergy_electricvehicles.csv'):

        if sourceType == 'api':
            sourceLoc = pathlib.Path(apiIntegrations.__file__).parent
            assert source in [x.stem for x in sourceLoc.iterdir()], f"Source must be valid api if source type api chosen. {source} not in {[x.stem for x in sourceLoc.iterdir()]}."
            logging.info(f"Data sourced from api {source}.")
        else:
            logging.info(f"Other data source type used: {sourceType}")

        self.source = source
        self.sourceType = sourceType

        self.location = pathlib.Path(filePath)
        assert self.location.exists(), "Filepath to data source does not exist."
        assert self.location.suffix == '.csv', "Must load data from csv."

        data = pd.read_csv(self.location, index_col=0, usecols=apiIntegrations.utilities.getApiColsOfInterest(self.source))
        cleanTextColName = 'clean'

        # Check data is clean
        try:
            assert all(topicmodelling.utilities.clean.isHygienic(text) for text in data[cleanTextColName])
        except KeyError:
            print("Column 'clean' not found. Please enter name of clean text column you would like to select:")
            cleanTextColName = input("-->")
            assert all(topicmodelling.utilities.clean.isHygienic(text) for text in data[cleanTextColName])
            logging.info(f"Column {cleanTextColName} identified by user as containing clean text data.")
        except AssertionError:
            logging.debug("Failed to add data due to data cleanliness issues. Refer to module topicmodelling.utilities.clean.")
            raise AssertionError("Document text must be clean before storing in corpus.")

        logging.info("Adding clean data to corpus.")
        self.data = data
        return

    def getBagsOfWords(self):
        """
        Convert column of text to column of bag of words.
        :return: list of words
        """
        assert 'text' in self.data.columns, "Column 'text' required in corpus data table."
        stopwords = nltk.corpus.stopwords.words('english')
        self.data['bow'] = self.data['text'].apply(lambda x: sorted(set(word for word in x if word not in stopwords)))
        return

    def updateVocabulary(self):
        assert isinstance(self.data, pd.DataFrame), "No data present."

        # Check bag of words has been created for each document. If not, try to create now.
        try:
            assert 'bow' in self.data.columns
        except AssertionError:
            logging.warning("Bag of words must be generated for each document first before generating the corpus vocabulary. Trying to generate now.")
            self.getBagsOfWords()
            logging.info("Bag of words generated during update vocabulary since no column 'bow' existed in data.")

        self.vocabulary = sorted(set(itertools.chain.from_iterable(self.data['bow'])))
        return





