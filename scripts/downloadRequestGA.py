import apiIntegrations.ga
import apiIntegrations.utilities

token = apiIntegrations.ga.getToken()
headers = apiIntegrations.ga.getHeaders(token)
urlStore = apiIntegrations.ga.getRequestURLStore()

requestSelection = 'Pancakes taste great also on Mondays...'

while requestSelection not in urlStore.keys():
    print(f"Type the request you want to make from {urlStore.keys()}")
    requestSelection = input("--> ")

requestURLs = apiIntegrations.ga.getRequestURLs(requestSelection)

data = apiIntegrations.ga.getRequestData(requestURLs, headers)

apiIntegrations.utilities.depositData(data, apiIntegrations.ga.__name__, requestSelection)


