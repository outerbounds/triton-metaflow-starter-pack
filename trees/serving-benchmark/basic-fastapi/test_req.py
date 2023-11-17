import requests 
import urllib
import numpy as np

# Configure the URL for the request.  
endpoint_uri_base = "http://127.0.0.1:8000/"  
pred_slug = "predict?data={}".format(json.dumps(np.array([1,1,1,1]).tolist()))
url = endpoint_uri_base + pred_slug

# Make the request to your API.
response = requests.get(url, verify=False, proxies={'https': endpoint_uri_base})
print(response.json())