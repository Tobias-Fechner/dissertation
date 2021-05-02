import topicmodelling.utilities.cleaning
import topicmodelling.utilities.evaluating
import topicmodelling.utilities.plotting
import apiIntegrations.utilities

import pandas as pd
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

import pathlib, logging
import spacy, gensim, tomotopy

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
        self.topics = pd.DataFrame()

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

    def updateDictionary(self):
        self.dictionary = gensim.corpora.dictionary.Dictionary(self.data.tokensProcessed)

        # Filter out words that occur in less than 3 documents, or more than 60% of the documents.
        self.dictionary.filter_extremes(no_below=3, no_above=0.6)
        return

    def checkDataClean(self):

        textColName = 'text'

        # Check data is clean
        try:
            assert all(topicmodelling.utilities.cleaning.isHygienic(text) for text in self.raw[textColName])
        except KeyError:
            print(
                f"Required column '{textColName}' not found. Please identify the column containing clean text data: \n{self.raw.columns}")
            textColName = input("-->")
            assert all(topicmodelling.utilities.cleaning.isHygienic(text) for text in self.raw[textColName])
            logging.info(f"Column {textColName} identified by user as containing clean text data.")
        except AssertionError:
            logging.debug(
                "Failed to add data due to data cleanliness issues. Refer to module topicmodelling.utilities.clean.")
            raise AssertionError("Document text must be clean before storing in corpus.")

        return

    def prepareData(self, texts):
        assert isinstance(texts, pd.Series)

        tokens = getTokens(texts)
        tokensProcessed = topicmodelling.utilities.cleaning.cleanTokens(tokens)

        self.updateDictionary()
        dtm = tokensProcessed.apply(lambda x: self.dictionary.doc2bow(x))

        self.data = pd.DataFrame(index=texts.index, data={
            'texts': texts,
            'tokens': tokens,
            'tokensProcessed': tokensProcessed,
            'length': tokensProcessed.apply(len),
            'dtm': dtm
        })
        return

    def getTopicsTable(self, topicmodel, numWords=10):
        assert isinstance(topicmodel, TopicModel), "Should pass in an instance of our own customer TopicModel class, not an actual model."
        keywords = []
        explainedProbs = []
        tokenPDs = []

        for topicID in range(topicmodel.num_topics):
            topic = topicmodel.getTopic(int(topicID), numWords=numWords)
            topWords = [n[0] for n in topic]
            # noinspection PyUnresolvedReferences
            explainedProb = np.sum([n[1] for n in topic])
            keywords.append(topWords)
            explainedProbs.append(explainedProb)
            tokenPDs.append(topicmodel.applyTopicTokenPD(self))

        self.topics = pd.DataFrame(index=range(topicmodel.num_topics), data={
            'keywords': keywords,
            'explainedProb': explainedProbs,
            'tokenPDs': tokenPDs,
        })
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
    def _getDocumentTopicPD(self, dtm):
        pass

    @abstractmethod
    def _getTopicTokenPD(self, corpus):
        pass

    @abstractmethod
    def instantiateModel(self, corpus):
        pass

    @abstractmethod
    def getTopic(self, topicID, numWords=10):
        pass

    def applyDominantTopics(self, corpus):
        if not 'topicPD' in corpus.data.columns:
            print("Missing topic probability distributions. Generating column now.")
            corpus = self.applyDocumentTopicPD(corpus)

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
            corpus = self.applyDocumentTopicPD(corpus)

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
            assert len(corpus.topics) != 0
        except AssertionError:
            print("No topics table found. Creating now.")
            corpus.getTopicsTable(self)

        if 'strongestDoc' in corpus.data.columns:
            print("Strongest document already in topics data.")
            return corpus
        else:
            corpus.topics.insert(len(corpus.topics.columns),
                                 'strongestDoc',
                                 pd.Series(range(len(corpus.topics))).apply(lambda x: self._getDominantDocument(corpus.data.topicPD, x)))
            return corpus

    def applyDocumentTopicPD(self, corpus):
        if not 'dtm' in corpus.data.columns:
            print("Data incomplete. Must have column 'dtm'.")
            return
        else:
            corpus.data.insert(len(corpus.data.columns),
                               'topicPD',
                               corpus.data.dtm.apply(lambda x: self._getDocumentTopicPD(x)))
            return corpus

    def applyTopicTokenPD(self, corpus):
        if 'tokenPD' in corpus.topics.columns:
            print("Token PD already in data.")
            return
        else:
            corpus.topics.insert(len(corpus.topics.columns),
                                 'tokenPD',
                                 self._getTopicTokenPD(corpus))
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

    def _getDocumentTopicPD(self, dtm):
        return self.model.get_document_topics(dtm, minimum_probability=0.005)

    # noinspection PyUnresolvedReferences
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

    def getTopic(self, topicID, numWords=10):
        return self.model.show_topic(topicID, topn=numWords)

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

        self.num_topics = len(self.model.get_topics())
        return

    def _getDocumentTopicPD(self, dtm):
        return self.model[dtm]

    # noinspection PyUnresolvedReferences
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

    def getTopic(self, topicID, numWords=10):
        return self.model.show_topic(topicID, topn=numWords)

class TomotopyHDP(TopicModel):
    def __init__(self, modelName):
        super().__init__(modelName)

    def instantiateModel(self, corpus, gamma=1, alpha=0.3, initial_k=100):
        self.model = tomotopy.HDPModel(min_cf=5, rm_top=10, gamma=gamma, alpha=alpha, initial_k=initial_k)
        self.num_topics = len(self.model.get_count_by_topics())
        for doc in corpus.data.tokensProcessed:
            self.model.add_doc(doc)

        self.model.burn_in = 100
        self.model.train(0)
        print('Removed top words:', self.model.removed_top_words)
        return

    def _getDominantTopics(self, topicPD):
        pass

    def _getSingleDominantTopic(self, topicPD):
        pass

    def _getDominantDocument(self, corpus, wantedTopic):
        pass

    def _getDocumentTopicPD(self, dtm):
        pass

    def getTopic(self, topicID, numWords=10):
        topic = self.model.get_topic_words(topicID, top_n=numWords)


    def train(self, corpus, iterations=1000, chunkSize=100, evalWith='c_v', evaluate=False, printDuring=False):
        assert evalWith in ('u_mass', 'c_v', 'c_uci', 'c_npmi'), "Must specify a coherence measure from the list ('u_mass', 'c_v', 'c_uci', 'c_npmi')."

        tracking = ['Number of Topics', 'Log Likelihood']
        if evaluate:
            tracking.append('Coherence Score')

        results = {}
        for tr in tracking:
            results[tr] = []

        corpus.updateDictionary()
        corpus.getTopicsTable(self, numWords=40)

        for chunk in tqdm(np.full((int(iterations/chunkSize),), chunkSize, dtype=int)):

            self.model.train(chunk)

            if evaluate:
                cohScores = topicmodelling.utilities.evaluating.scoreModelTomotopy(corpus, cohScoreNames=[evalWith])
                results['Coherence Score'].append(cohScores[evalWith])

            results['Number of Topics'].append(self.model.live_k)
            results['Log Likelihood'].append(self.model.ll_per_word)

            if printDuring:
                print(results)

        self.num_topics = self.model.live_k

        return results

    @staticmethod
    def _getTokenIDPD(tokenPDs, corpus):
        return [(corpus.dictionary.token2id[token], PD) for token, PD in tokenPDs]

    def _getTopicTokenPD(self, corpus):
        termPDs = corpus.topics.index.to_series().apply(lambda x: self.getTopic(x, numWords=40))
        return termPDs.apply(lambda x: self._getTokenIDPD(x, corpus))


