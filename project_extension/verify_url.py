import urllib.request
import sys

url = 'https://github.com/jcssilva4/deep_learning_proj1/blob/master/models/10_20_20191231-22_36_34_ep_5000_encoder_model.pth?raw=true'

try:
    with urllib.request.urlopen(url) as response:
        print(f"Status: {response.status}")
        print(f"URL: {response.geturl()}")
        content = response.read(100) # Read first 100 bytes
        print(f"First 100 bytes: {content}")
except Exception as e:
    print(f"Error: {e}")
