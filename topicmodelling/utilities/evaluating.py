from gensim.models import CoherenceModel
from tqdm import tqdm
import tomotopy
import numpy as np

def scoreModelGensim(topics, texts, dictionary, cohScoreNames=('u_mass', 'c_v', 'c_uci', 'c_npmi')):
    scores = {}

    for cohScoreName in tqdm(cohScoreNames):
        scores[cohScoreName] = CoherenceModel(topics=topics,
                                              texts=texts,
                                              dictionary=dictionary,
                                              coherence=cohScoreName).get_coherence()

    return scores

def scoreModelTomotopy(corpus, cohScoreNames=('u_mass', 'c_v', 'c_uci', 'c_npmi')):
    scores = {}

    tempCorpus = tomotopy.utils.Corpus()
    _ = [tempCorpus.add_doc(doc) for doc in corpus.data.tokensProcessed]

    for cohScoreName in tqdm(cohScoreNames):
        coh = tomotopy.coherence.Coherence(tempCorpus,
                                           targets=[w[1] for w in corpus.dictionary.items()],
                                           coherence=cohScoreName)

        scores[cohScoreName] = np.mean(corpus.topics.keywords.apply(lambda x: coh.get_score(x)))

    return scores