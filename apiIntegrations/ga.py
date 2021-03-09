from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import json
import logging

URL_STORE = {
    'channels': "https://www.getabstract.com/api/library/v2/channels?activeOnly=true&language={}&page={}&psize={}",
    'channelsSearch': "https://www.getabstract.com/api/library/v2/channels/search?language={}&page={}&psize={}&query={}",
    'summaries': "https://www.getabstract.com/api/library/v2/summaries?activeOnly=true&excludeReviews=false&globalRightsOnly=false&sorting=latest&language={}&page={}&psize={}",
}

def getConfigDetails():
    """
    Reads in config.json file, returns three config details.
    :return: token url, client ID, and client secret for getAbstract API.
    """

    ## Get API config details containing client ID and secret
    con_file = open(r"C:\Users\Tobias Fechner\Documents\1_Uni\fyp\git_repo_fyp\apiConfig.json")
    config = json.load(con_file)
    con_file.close()

    # Set client ID and secret
    client_id = config['clientID_getAbstract']
    client_secret = config['clientSecret_getAbstract']
    tokenURL = config['tokenURL_getAbstract']

    return tokenURL, client_id, client_secret

def __urlBuilder(selection, params):
    """
    Takes in the user input selection of which request URL they would like to generate, and a tuple of parameters to provide to build the URL.
    :param selection: user input of desired request URL
    :param params: parameters required to format the URL
    :return: request URL
    """

    if selection == 'options':
        return list(URL_STORE.keys())
    else:
        assert URL_STORE[selection].count('{}') == len(params), "Request URL selected not compatible with number of parameters provided."

        if 'search' in selection or 'Search' in selection:
            queries = params.pop()
            urls = [URL_STORE[selection].format(*params, query.replace(' ', '+')) for query in queries]
        else:
            urls = [URL_STORE[selection].format(*params)]
        return tuple(urls)

def getRequestURL(language='en', pageSize=1000, pageNum=0):
    """
    Function used to generate a valid request URL by combining input parameters with one of the unformatted URL strings from the URL_STORE
    :param language: document language
    :param pageSize: number of records returned for one page
    :param pageNum: page number of search results
    :param query: string used in search
    :return: valid request URL
    """

    try:

        print(f"Type the request you want to make from {URL_STORE.keys()}")
        selection = input("--> ")
        assert selection in URL_STORE.keys()

    except AssertionError:

        print(f"You didn't select one of these: {URL_STORE.keys()}")
        print("Type the request you want to make: ")
        selection = input("--> ")
        assert selection in URL_STORE.keys(), "Yo, why are you not typing one of the ones listed?..."

    if 'search' in selection or 'Search' in selection:
        queries = [0]
        print("Please enter the query terms you would like to search for, separated by new lines. These will be searched for with consecutive queries:")
        while queries[-1] != "":
            queries.append(input("-->"))
        queries.pop(0)
        queries.pop()
    else:
        queries = None

    params = []
    for p in [language, pageNum, pageSize, queries]:
        params.append(p) if p is not None else logging.info(f"No parameter received for {p}")

    urls = __urlBuilder(selection, params)

    return urls, selection

def getToken():

    tokenURL, clientID, clientSecret = getConfigDetails()

    client = BackendApplicationClient(client_id=clientID)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(token_url=tokenURL, client_id=clientID, client_secret=clientSecret)

    return token

