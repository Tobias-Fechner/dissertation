import networkx as nx
import plotly.graph_objects as go

def getLayoutFunc(layout='spring'):

    layoutGenerator = {
        'spring': lambda x: nx.spring_layout(x),
        'circular': lambda x: nx.circular_layout(x),
        'kamada': lambda x: nx.kamada_kawai_layout(x),
        'planar': lambda x: nx.planar_layout(x),
        'random': lambda x: nx.random_layout(x),
        'shell': lambda x: nx.shell_layout(x),
        'spectral': lambda x: nx.spectral_layout(x),
        'spiral': lambda x: nx.spiral_layout(x),
        'multipartite': lambda x: nx.multipartite_layout(x),
    }

    assert layout in layoutGenerator.keys(), f"Passed layout string not in {layoutGenerator.keys()}"
    return layoutGenerator[layout]

def makeEdgeTrace(x, y, text, width, edgeColour):
    return go.Scatter(x=x, y=y, line=dict(width=width, color=edgeColour), hoverinfo='text', text=([text]), mode='lines')

def getEdgeTraces(g, pos_, edgeColour='Aquamarine', weightCorrectionFactor=1.5, weightCorrectionExponent=0.7):
    edgeTraces = []
    for edge in g.edges():
        if g.edges()[edge]['weight'] > 0:
            source = edge[0]
            target = edge[1]

            x0, y0 = pos_[source]
            x1, y1 = pos_[target]

            text = f"{source} --> {target}: {str(g.edges()[edge]['weight'])}"

            trace = makeEdgeTrace([x0, x1, None],
                                  [y0, y1, None],
                                  text,
                                  edgeColour=edgeColour,
                                  width=weightCorrectionFactor * g.edges()[edge]['weight'] ** weightCorrectionExponent)

            edgeTraces.append(trace)
        else:
            continue

    return edgeTraces

def getNodeTrace(g, pos_, nodeSizeCorrection=0.25, nodeColour='cornflowerblue', nodeTextSize=10):
    nodeTrace = go.Scatter(x=[], y=[], text=[],
                           textposition="top center",
                           textfont=dict(family="Arial", size=nodeTextSize),
                           mode='markers+text',
                           hoverinfo='none',
                           marker=dict(color=[], size=[], line=None))

    # For each node in graph, get the position and size and add to the node_trace
    for node in g.nodes():
        x, y = pos_[node]
        nodeTrace['x'] += tuple([x])
        nodeTrace['y'] += tuple([y])
        nodeTrace['marker']['color'] += tuple([nodeColour])
        nodeTrace['marker']['size'] += tuple([nodeSizeCorrection * g.nodes()[node]['degree']])
        nodeTrace['text'] += tuple(['<b>' + str(node) + '</b>'])

    return nodeTrace

def getFigure(g, layout='spring', nodeSizeCorrection=0.25, nodeColour='cornflowerblue', nodeTextSize=10,
              edgeColour='Aquamarine', weightCorrectionFactor=1.5, weightCorrectionExponent=0.7):
    f = getLayoutFunc(layout)
    pos_ = f(g)

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',  # transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # transparent 2nd background
        xaxis={'showgrid': False, 'zeroline': False},  # no gridlines
        yaxis={'showgrid': False, 'zeroline': False},  # no gridlines
    )

    # Create figure
    fig = go.Figure(layout=layout)

    # Add all edge traces
    for trace in getEdgeTraces(g, pos_, edgeColour=edgeColour, weightCorrectionFactor=weightCorrectionFactor,
                               weightCorrectionExponent=weightCorrectionExponent):
        fig.add_trace(trace)

    # Add node trace
    fig.add_trace(getNodeTrace(g, pos_, nodeSizeCorrection=nodeSizeCorrection,
                               nodeColour=nodeColour, nodeTextSize=nodeTextSize))

    # Remove legend
    fig.update_layout(showlegend=False)
    # Remove tick labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig