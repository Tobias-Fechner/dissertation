import nltk
import gensim
import pathlib
import pandas as pd
import apiIntegrations.utilities
import apiIntegrations.ga
import topicmodelling.utilities.clean
import topicmodelling.classes

stopwords = nltk.corpus.stopwords.words('english')

# Load data
dataDir = pathlib.Path(pathlib.Path(r"C:\Users\Tobias Fechner\Documents\1_Uni\fyp\git_repo_fyp\data"), 'apiIntegrations.ga')

print("Available files: ")
availableFiles = [(count, f"{count}: {str(filePath.stem)}") for count, filePath in enumerate(dataDir.iterdir())]
print("\n".join(list(zip(*availableFiles))[1]))

selection = '-1'
while int(selection) not in list(zip(*availableFiles))[0]:
    print("Specify file to select by number (esc to break):")
    selection = input("--> ")
    if selection == 'esc':
        selection = ''
        break

df = pd.read_csv(list(dataDir.iterdir())[int(selection)], index_col=0)
print(df.head(2))

# Pre-Process
wantedCols = apiIntegrations.utilities.getApiColsOfInterest('ga')
data = df.loc[:, wantedCols]
data['combinedIntroContent'] = apiIntegrations.ga.addIntroToContent(data)
data['text'] = topicmodelling.utilities.clean.cleanHTML(data['combinedIntroContent'])

# Create corpus, get bag of words and generate vocabulary
corpus = topicmodelling.classes.Corpus('test')
corpus.data = data
corpus.getBagsOfWords()
corpus.updateDictionary()
print(f"""Generated vocabulary: {"', '".join(corpus.dictionary[:5])}...""")

# Generate LDA model
dictionary = gensim.corpora.Dictionary(corpus.data['bow'])
tfMatrix = [dictionary.doc2bow(text) for text in corpus.data['bow']]
lda = gensim.models.ldamodel.LdaModel(corpus=tfMatrix, num_topics=20, iterations=100)



