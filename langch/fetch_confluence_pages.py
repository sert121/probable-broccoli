# This code sample uses the 'requests' library:
# http://docs.python-requests.org
import requests
from requests.auth import HTTPBasicAuth
import json
import os

#set confluence URL
url = os.getenv("URL")

# set confluence auth creds from env vars
email = os.getenv("AUTH_EMAIL")
auth_token = os.getenv("AUTH_TOKEN")

auth = HTTPBasicAuth(email, auth_token)

headers = {
  "Accept": "application/json"
}

response = requests.request(
   "GET",
   url,
   headers=headers,
   auth=auth
)

print(json.dumps(json.loads(response.text), sort_keys=True, indent=4, separators=(",", ": ")))