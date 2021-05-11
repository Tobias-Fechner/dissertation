from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import json
import logging
import pandas as pd
import http.client
import math
from tqdm import tqdm
import ast

def getRequestURLStore():
    urlStore = {
        'channels': "https://www.getabstract.com/api/library/v2/channels?activeOnly=true&language={}&page={}&psize={}",
        'channelsSearch': "https://www.getabstract.com/api/library/v2/channels/search?language={}&page={}&psize={}&query={}",
        'summaries': "https://www.getabstract.com/api/library/v2/summaries?activeOnly=true&excludeReviews=false&globalRightsOnly=false&sorting=latest&language={}&page={}&psize={}",
    }
    return urlStore

def getHeaders(token):
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'authorization': 'Bearer ' + token['access_token'],
    }
    return headers

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

    urlStore = getRequestURLStore()

    if selection == 'options':
        return list(urlStore.keys())
    else:
        assert urlStore[selection].count('{}') == len(params), "Request URL selected not compatible with number of parameters provided."

        if 'search' in selection or 'Search' in selection:
            queries = params.pop()
            urls = [urlStore[selection].format(*params, query.replace(' ', '+')) for query in queries]
        else:
            urls = [urlStore[selection].format(*params)]
        return tuple(urls)

def getRequestURLs(requestSelection, language='en', pageSize=1000, pageNum=0):
    """
    Function used to generate a valid request URL by combining input parameters with one of the unformatted URL strings from the urlStore
    :param requestSelection: request url to be generated.
    :param language: document language
    :param pageSize: number of records returned for one page
    :param pageNum: page number of search results
    :return: valid request URL
    """

    if 'search' in requestSelection or 'Search' in requestSelection:
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

    urls = __urlBuilder(requestSelection, params)

    return urls

def getToken():

    tokenURL, clientID, clientSecret = getConfigDetails()

    client = BackendApplicationClient(client_id=clientID)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(token_url=tokenURL, client_id=clientID, client_secret=clientSecret)

    return token

def getChannelsMask(wantedChannels, channels, languages):

    assert isinstance(channels, pd.Series) and isinstance(languages, pd.Series)

    try:
        channelIDs = channels.str.extract(r"\'id\': (\d\d\d?\d?\d?\d?),", expand=False)
        assert not all(channelIDs.isna())
    except AssertionError:
        channelIDs = channels.astype(str).str.extract(r"'id': (\d\d\d?\d?\d?\d?),", expand=False)
        assert not all(channelIDs.isna()), "Failed to retrieve channel ID from summaries dataframe."

    if not len(channelIDs) > 0:
        logging.info("No channels of interest found. Returning None.")
        return None
    else:
        pass

    # Cast elements to string and join with OR operator. Then search for any of the channel IDs in the summaries channel IDs
    maskChannels = channelIDs.str.fullmatch('|'.join([str(c) for c in wantedChannels]))
    maskLanguage = languages.str.fullmatch('en')

    return maskChannels & maskLanguage

def getRequestResponse(requestURL, headers, field=None):
    conn = http.client.HTTPSConnection("getabstract.com")
    conn.request("GET", requestURL, headers=headers)

    response = conn.getresponse()
    dataByte = response.read().decode('utf-8')
    dataJSON = json.loads(dataByte)

    # Check if response field is specified. If not, return full response. If it is specified but not present in response, ask for field to be re-specified.
    if field is None:
        return dataJSON

    else:
        assert isinstance(field, str), "Desired field must be given as string."

        try:
            assert field in dataJSON.keys()

        except AssertionError:
            selection = 'a'
            count = 1
            while selection not in dataJSON.keys() and count <4:
                print(f"(Attempt {count}) Please select a field to return from the list: \n{dataJSON.keys()}")
                input("-->")
                count += 1

            if selection not in dataJSON.keys():
                raise ValueError("Desired field not present in response dictionary.")
            else:
                return dataJSON[selection]

        return dataJSON[field]

def getRequestData(requestURLs, headers, field='items'):
    data = pd.DataFrame([])

    for url in requestURLs:
        res = getRequestResponse(url, headers, field=field)
        data = data.append(pd.DataFrame(res).set_index('id'))

    return data

def getAllSummaries(sample=False):
    token = getToken()
    headers = getHeaders(token)

    summariesURL = getRequestURLs('summaries', pageSize=1)[0]
    summariesCount = getRequestResponse(summariesURL, headers, 'countTotal')

    pageSize = 1000
    if sample:
        pages = 1
    else:
        pages = int(math.floor(summariesCount/ pageSize))

    data = pd.DataFrame([])

    for pageNum in tqdm(range(pages + 1)):
        url = getRequestURLs('summaries', pageSize=pageSize, pageNum=pageNum)
        data = data.append(getRequestData(url, headers))

    return data

def addIntroToContent(data):
    return data.apply(lambda x: x['introHtml'] + x['contentHtml'], axis=1)

def extractYearPublished(src):
    srcConverted = src.apply(lambda x: ast.literal_eval(x))
    return srcConverted.apply(lambda x: x['year'])


