import pandas as pd
import gensim
import numpy as np

def getHellingerDistances(row, df, maxDistance=0.25):
    otherPDs = df.loc[df.index != row.name, 'tokenPDs']
    distances = otherPDs.apply(lambda x: gensim.matutils.hellinger(row['tokenPDs'], x)).sort_values()
    return list(zip(distances.loc[distances < maxDistance].index, distances.loc[distances < maxDistance]))

def getNodesEdges(corpus, onlyConnectedTopics=True):
    if onlyConnectedTopics:
        maskConnected = corpus.topics.degree != 0
    else:
        maskConnected = np.ones_like(corpus.topics.degree, dtype=bool)

    # Combine all filters (currently only one) into one mask and apply
    nodes = corpus.topics.loc[maskConnected]
    print(f"Dropped the following topics as they were unconnected: {corpus.topics.loc[~maskConnected].index}")

    nodePairs = nodes.apply(lambda x: [(x.name, topicID, distance) for topicID, distance in x['topicDistances']], axis=1)
    edges = pd.concat([pd.DataFrame(pair, columns=['source', 'target', 'weight']) for pair in nodePairs], ignore_index=True)

    # noinspection PyUnresolvedReferences
    assert np.all(np.unique(nodes.index) == np.unique(edges.source)), "Mis-match in nodes and edges lists."
    return nodes, edges


