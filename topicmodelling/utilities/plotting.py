import plotly.graph_objects as go
import numpy as np
import pandas as pd

def _plotScatter(xData, yData, mode, xTitle="[X Axes Tile]", yTitle="[Y Axes Title]"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xData,
        y=yData,
        mode=mode,
    ))

    fig.update_xaxes(title_text=xTitle)
    fig.update_yaxes(title_text=yTitle)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=20,
            color="black"
    ))
    return fig

def plotScatter(xData, yData, xTitle="[X Axes Tile]", yTitle="[Y Axes Title]"):
    return _plotScatter(xData, yData, 'markers', xTitle=xTitle, yTitle=yTitle)

def plotLine(xData, yData, xTitle="[X Axes Tile]", yTitle="[Y Axes Title]"):
    return _plotScatter(xData, yData, 'lines+markers', xTitle=xTitle, yTitle=yTitle)

def plotBar(xData, yData, xTitle="[X Axes Tile]", yTitle="[Y Axes Title]"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xData,
        y=yData,
    ))

    fig.update_xaxes(title_text=xTitle)
    fig.update_yaxes(title_text=yTitle)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=20,
            color="black"
    ))
    return fig

def plotDocCountYearPublished(yearsData, minYear=None, maxYear=None):
    print(f"Dataset length before applying min/ max year mask: {len(yearsData)}.")
    if minYear is None:
        maskMin = yearsData >= yearsData.min()
    else:
        maskMin = yearsData >= minYear
    if maxYear is None:
        maskMax = yearsData <= yearsData.max()
    else:
        maskMax = yearsData <= maxYear

    # noinspection PyUnresolvedReferences
    docCountInYear = [np.sum(yearsData == yr) for yr in np.unique(yearsData.loc[maskMin & maskMax])]

    countDf = pd.DataFrame(data={'yearPublished': np.unique(yearsData.loc[maskMin & maskMax]), 'docCount': docCountInYear})
    print(f"Dataset length after applying min/ max year mask: {len(countDf)}")

    return plotBar(countDf.yearPublished, countDf.docCount, xTitle="Year Published", yTitle="Document Count")

def plotDocLengths(docs, minLength=None, maxLength=None):
    lengths = docs.apply(len)
    print(f"Dataset length before applying min/ max length mask: {len(lengths)}")

    if minLength is None:
        maskMin = lengths >= lengths.min()
    else:
        maskMin = lengths >= minLength
    if maxLength is None:
        maskMax = lengths <= lengths.max()
    else:
        maskMax = lengths <= maxLength

    print(f"Dataset length after applying min/ max length mask: {len(lengths.loc[maskMin & maskMax])}")

    return plotScatter(lengths.loc[maskMin & maskMax].index,
                       lengths.loc[maskMin & maskMax].values,
                       xTitle="Document ID",
                       yTitle="Document Token Count")

def getTopicPies(corpus, numberOfDocs=5):
    dfMeta = corpus.raw.loc[:, ['title', 'author']]
    dfMeta['docLength'] = corpus.data.tokensProcessed.apply(len)

    figs=[]

    for index, row in corpus.data.iloc[:numberOfDocs, :].iterrows():
        df = pd.DataFrame(data=row.dominantTopics, columns=['topicID', 'prob', 'keywords'])
        df = df.append(pd.DataFrame(['other', 1 - sum(df.prob), 'NA'], index=df.columns).T, ignore_index=True)
        fig = go.Figure(
            data=[go.Pie(
                labels=df.topicID,
                values=df.prob,
                text=df.keywords,
                hole=.3,
                hovertemplate="Topic ID: %{label}<br>Probability: %{percent}<br>Keywords: %{text}",
                textinfo='percent',
                sort=False
            )]
        )
        fig.update_layout(
            title_text=f"Name: {dfMeta.loc[index, 'title']}<br>Author: {dfMeta.loc[index, 'author']}<br>Length: {dfMeta.loc[index, 'docLength']}",
            legend=dict(title='Topic ID'))
        figs.append(fig)

    return figs

def plotModelEvalComparison(nameModelA, scoresModelA, nameModelB, scoresModelB):
    fig = go.Figure(data=[
        go.Bar(name=f'{nameModelA}', x=list(scoresModelA.keys()), y=list(scoresModelA.values())),
        go.Bar(name=f'{nameModelB}', x=list(scoresModelB.keys()), y=list(scoresModelB.values()))
    ])
    fig.update_layout(barmode='group')
    fig.update_layout(
        font=dict(
            family="Arial",
            size=20,
            color="black"
    ))
    return fig





