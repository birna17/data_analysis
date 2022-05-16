from urllib.request import urlopen
import numpy as np
import pandas as pd
import requests
import json


ds = requests.get('https://api.datacite.org/dois/application/vnd.datacite.datacite+json/10.12753/2066-026x-19-008')

