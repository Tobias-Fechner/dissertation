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
        self.topics = None

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

    def instantiateTopicsTable(self, model):
        keywords = []

        for topicID in range(model.num_topics):
            topWords = [n[0] for n in model.show_topic(int(topicID), topn=5)]
            keywords.append(topWords)

        self.topics = pd.DataFrame(index=range(model.num_topics), data={'keywords': keywords})
        return

class TopicModel(ABC):
    def __init__(self, modelName):
        self.modelName = modelName
        self.model = None
        self.num_topics = None

    @abstractmethod
    def _getDominantTopics(self, topicPD):
        pass

    @abstractmethod
    def _getSingleDominantTopic(self, topicPD):
        pass

    @abstractmethod
    def _getDominantDocument(self, corpus, wantedTopic):
        pass

    @abstractmethod
    def _getTopicProbabilityDistributions(self, dtm):
        pass

    @abstractmethod
    def instantiateModel(self, corpus):
        pass

    def applyDominantTopics(self, corpus):
        if not 'topicPD' in corpus.data.columns:
            print("Missing topic probability distributions. Generating column now.")
            corpus = self.applyTopicProbabilityDistributions(corpus)

        if 'dominantTopics' in corpus.data.columns:
            print("Dominant topics already in corpus data.")
            return corpus
        else:
            corpus.data.insert(len(corpus.data.columns),
                               'dominantTopics',
                               corpus.data.topicPD.apply(lambda x: self._getDominantTopics(x)))
            return corpus

    def applySingleDominantTopic(self, corpus):
        if not 'topicPD' in corpus.data.columns:
            print("Missing topic probability distributions. Generating column now.")
            corpus = self.applyTopicProbabilityDistributions(corpus)

        if 'singleDominantTopic' in corpus.data.columns:
            print("Single dominant topic already in corpus data.")
            return
        else:
            corpus.data.insert(len(corpus.data.columns),
                               'singleDominantTopic',
                               corpus.data.topicPD.apply(lambda x: self._getSingleDominantTopic(x)))
            return corpus

    def applyDominantDocument(self, corpus):
        try:
            assert corpus.topics
        except AssertionError:
            print("No topics table found. Creating now.")
            corpus.instantiateTopicsTable(self)

        if 'strongestDoc' in corpus.data.columns:
            print("Strongest document already in topics data.")
            return corpus
        else:
            corpus.topics.insert(len(corpus.topics.columns),
                                 'strongestDoc',
                                 pd.Series(range(len(corpus.topics))).apply(lambda x: self._getDominantDocument(corpus.data.topicPD, x)))
            return corpus

    def applyTopicProbabilityDistributions(self, corpus):
        if not 'dtm' in corpus.data.columns:
            print("Data incomplete. Must have column 'dtm'.")
            return
        else:
            corpus.data.insert(len(corpus.data.columns),
                               'topicPD',
                               corpus.data.dtm.apply(lambda x: self._getTopicProbabilityDistributions(x)))
            return corpus

class LDA(TopicModel):
    def __init__(self):
        super().__init__('LDA')

    def instantiateModel(self, corpus, num_topics=20):
        try:
            assert 'dtm' in corpus.data.columns
        except AssertionError:
            print("Corpus data incomplete. Missing 'dtm' column.")
            return

        self.model = gensim.models.ldamodel.LdaModel(corpus=corpus.data.dtm,
                                                     id2word=corpus.dictionary,
                                                     num_topics=num_topics,
                                                     random_state=100,
                                                     update_every=1,
                                                     chunksize=20,
                                                     passes=50,
                                                     alpha='symmetric',
                                                     iterations=100,
                                                     per_word_topics=True)

        self.num_topics = self.model.num_topics
        return

    def _getTopicProbabilityDistributions(self, dtm):
        return self.model.get_document_topics(dtm, minimum_probability=0.005)

    def _getDominantTopics(self, topicPD, threshold=0.9):
        # Sort topics by greatest probabilities
        sortedByProb = np.array(sorted(topicPD, key=lambda x: x[1], reverse=True))

        # Generate cumulative sum of topic probabilities
        cs = np.cumsum([i[1] for i in sortedByProb], axis=0)

        # Return list of dominant topics
        dominants = sortedByProb[:np.argmax(cs>threshold)+1]

        result = []

        for topicID, probability in dominants:
            topWords = [n[0] for n in self.model.show_topic(int(topicID), topn=6)]
            result.append((int(topicID), round(probability, 3), topWords))

        return result

    def _getSingleDominantTopic(self, topicPD):

        # Sort topics by greatest probabilities
        sortedByProb = sorted(topicPD, key=lambda x: x[1], reverse=True)

        return sortedByProb[0][0], sortedByProb[0][1]

    def _getDominantDocument(self, topicPDs, wantedTopic):
        wantedTopicProbs = topicPDs.apply(lambda x: [a[1] for a in x if a[0] == wantedTopic])
        if wantedTopicProbs.apply(len).sum() == 0:
            return
        else:
            nonZerosUnpacked = wantedTopicProbs[wantedTopicProbs.apply(len) != 0].apply(lambda x: x[0])
            return nonZerosUnpacked.idxmax(), nonZerosUnpacked.max()

class HDP(TopicModel):
    def __init__(self):
        super().__init__('HDP')

    def instantiateModel(self, corpus):
        try:
            assert 'dtm' in corpus.data.columns
        except AssertionError:
            print("Corpus data incomplete. Missing 'dtm' column.")
            return

        self.model = gensim.models.hdpmodel.HdpModel(corpus=corpus.data.dtm.to_list(),
                                                     id2word=corpus.dictionary,
                                                     random_state=50,
                                                     chunksize=30,
                                                     max_chunks=600)

        self.num_topics = len(self.model.print_topics(-1))
        return

    def _getTopicProbabilityDistributions(self, dtm):
        pass

    def _getDominantTopics(self, topicPD):
        pass

    def _getSingleDominantTopic(self, topicPD):
        pass

    def _getDominantDocument(self, corpus, wantedTopic):
        pass





