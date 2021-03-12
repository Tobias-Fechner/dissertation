"""
Script uses getAbstract api to download summaries, based on key terms provided by the user. The data is cleaned and saved to disk.
"""

import apiIntegrations.ga
import apiIntegrations.utilities
import topicmodelling.utilities.clean

print("Getting token.")
token = apiIntegrations.ga.getToken()
headers = apiIntegrations.ga.getHeaders(token)

# Get request URLs to search for channels based on key words given as input
print("Getting channel request url.")
channelsSearchURLs = apiIntegrations.ga.getRequestURLs('channelsSearch')

# Get all channels using generated channel URLs
channels = apiIntegrations.ga.getRequestData(channelsSearchURLs, headers)

# Get all summaries accessible with token privileges
summaries = apiIntegrations.ga.getAllSummaries()

# Generate mask to select just channels of interest
mask = apiIntegrations.ga.getChannelsMask(channels.index, summaries['channels'], summaries['language'])

data = summaries.loc[mask]
data['textHtml'] = apiIntegrations.ga.addIntroToContent(data)
data['text'] = topicmodelling.utilities.clean.cleanHTML(data['textHtml'])

print("Please enter a filename for the data collection.")
fileName = input("-->")

apiIntegrations.utilities.depositData(data, apiIntegrations.ga.__name__, fileName)

