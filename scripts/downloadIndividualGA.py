import json
import apiIntegrations.ga
import http.client
import pandas as pd
import os
import pathlib
import apiIntegrations.utilities

token = apiIntegrations.ga.getToken()
requestURLs, selection = apiIntegrations.ga.getRequestURL()

headers = {
    'accept': 'application/json',
    'content-type': 'application/json',
    'authorization': 'Bearer ' + token['access_token'],
}

data = pd.DataFrame([])

for url in requestURLs:
    conn = http.client.HTTPSConnection("getabstract.com")
    conn.request("GET", url, headers=headers)

    response = conn.getresponse()
    dataByte = response.read().decode('utf-8')
    dataJSON = json.loads(dataByte)

    data = data.append(pd.DataFrame(dataJSON['items']).set_index('id'))

dataDir = pathlib.Path(os.getcwd(), 'data', 'getAbstract')

if not dataDir.exists():
    os.mkdir(dataDir)
else:
    pass

fileName = f"{selection}.csv"
filePath = pathlib.Path(dataDir, fileName)

apiIntegrations.utilities.depositData(data, filePath)

