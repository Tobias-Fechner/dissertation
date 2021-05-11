import networkx as nx
from operator import itemgetter
import networking.utilities
import networking.plotting

class Graph:
    def __init__(self, nodes, edges):
        self.g = nx.Graph()
        self.nodes = nodes
        self.edges = edges

        self.density = None
        self.isConnected = None
        self.largestComponent = None
        self.diameterLargestComponent = None
        self.triadicClosure = None
        self.topDegrees = None
        self.topBetweenness = None
        self.topEigenvector = None

        self.g.add_nodes_from(nodes.index)
        self.g.add_weighted_edges_from(list(zip(edges.source, edges.target, edges.weight)))
        self.nodes['attributes'] = nodes.apply(lambda x: self.getRowAttributes(x), axis=1)
        self.nodes.attributes.apply(lambda x: self.addAttributesToGraph(x))
        self.updateMetrics()
        self.updateDerivedAttributes()

    @staticmethod
    def getRowAttributes(row):
        name = row.name

        keywords = ('keywords', {name: row.iloc[0]})
        strongestDocTitle = ('strongestDocTitle', {name: row.iloc[2]})
        numConnectingTopics = ('numConnectingTopics', {name: row.iloc[-1]})

        return keywords, strongestDocTitle, numConnectingTopics

    def addAttributesToGraph(self, attributes):
        for attribute in attributes:
            nx.set_node_attributes(self.g, values=attribute[1], name=attribute[0])

    def updateDerivedAttributes(self):
        # noinspection PyCallingNonCallable
        degrees = dict(self.g.degree(self.g.nodes()))
        betweenness = nx.betweenness_centrality(self.g)
        eigenvector = nx.eigenvector_centrality(self.g)

        nx.set_node_attributes(self.g, degrees, 'degree')
        nx.set_node_attributes(self.g, betweenness, 'betweenness')
        nx.set_node_attributes(self.g, eigenvector, 'eigenvector')

        self.topDegrees = sorted(degrees.items(), key=itemgetter(1), reverse=True)[:20]
        self.topBetweenness = sorted(betweenness.items(), key=itemgetter(1), reverse=True)[:20]
        self.topEigenvector = sorted(eigenvector.items(), key=itemgetter(1), reverse=True)[:20]

    def updateMetrics(self):
        self.density = nx.density(self.g)
        self.isConnected = nx.is_connected(self.g)
        self.largestComponent = max(nx.connected_components(self.g), key=len)
        self.diameterLargestComponent = nx.diameter(self.g.subgraph(self.largestComponent))
        self.triadicClosure = nx.transitivity(self.g)

    def getShortestPath(self, source, target):
        sp = nx.shortest_path(self.g, source=source, target=target)
        print(f"Shortest path between {source} and {target} = {sp}")
        return sp

    def printGraph(self, layout='spring', nodeSizeCorrection=0.25, nodeColour='cornflowerblue', nodeTextSize=10,
              edgeColour='Aquamarine', weightCorrectionFactor=1.5, weightCorrectionExponent=0.7, show=True):

        fig = networking.plotting.getFigure(self.g, layout,
                                            nodeSizeCorrection=nodeSizeCorrection,
                                            nodeColour=nodeColour,
                                            nodeTextSize=nodeTextSize,
                                            edgeColour=edgeColour,
                                            weightCorrectionFactor=weightCorrectionFactor,
                                            weightCorrectionExponent=weightCorrectionExponent)

        if show:
            fig.show()
            return
        else:
            return fig




