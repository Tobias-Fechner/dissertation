import pybliometrics
import json


## Load configuration
con_file = open("../apiConfig.json")
config = json.load(con_file)
con_file.close()

## Initialize client
client = ElsClient(config['apikey'])

## Scopus (Abtract) document example
# Initialize document with ID as integer
scp_doc = AbsDoc(scp_id = 84872135457)
if scp_doc.read(client):
    print ("scp_doc.title: ", scp_doc.title)
    scp_doc.write()
else:
    print ("Read document failed.")


