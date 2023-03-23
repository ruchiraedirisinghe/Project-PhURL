'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Testing ####

'''

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import joblib
from urllib.parse import urlparse
from tld import get_tld
from urllib.parse import urlparse
from tld import get_tld
import os.path
from urllib.parse import urlparse

# Load model
model = joblib.load('lgb_model.joblib')


# Define feature extraction functions

## Feature 1: Check whether the URL has a IPV address or not 

def having_ip_address(url):
    match = re.search(
       
        # IPv4
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  
        
        # IPv4 in hexadecimal format
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' 
        
        # Ipv6
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  
    
    # print match group or No matching pattern found by scanning them.
    
    if match:
        return 1
    else:
        return 0

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    
    # print match group or No matching pattern found by scanning them.
    
    if match:
        return 1
    else:
        return 0

## Feature 2: Scan whether the URL is a Google Link

from googlesearch import search

def google_index(url):
    site = search(url, 5)
    return 1 if site else 0


# Scan how many dots the URL contains

def count_dot(url):
    count_dot = url.count('.')
    return count_dot


## Feature 3: Scan how many 'www's the URL contains

def count_www(url):
    url.count('www')
    return url.count('www')



## Feature 4: Scan how many '@'s the URL contains

def count_atrate(url):
     
    return url.count('@')



## Feature 5: Scan how many ' / 's the URL contains

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')


## Feature 6: Scan how many embeds the URL contains

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')



## Feature 7: Scan whether a URL shortening method has been used

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0
    
    

## Feature 8: Scan the protocol the web URL is using

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')



## Feature 9: Scan is there are any spaces contained in the URL

def count_per(url):
    return url.count('%')


def count_ques(url):
    return url.count('?')


def count_hyphen(url):
    return url.count('-')


def count_equal(url):
    return url.count('=')




## Feature 10: Scan the length of the URL

def url_length(url):
    return len(str(url))



## Feature 11: Scan the Hostname's length

def hostname_length(url):
    return len(urlparse(url).netloc)



## Feature 12: Scan for suspicious words inside the URL

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
    
    

## Feature 13: Scan the number of digits used

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits



## Feature 14: Scan the number of letters used

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters



## Feature 15: Scan the First Directory Length

def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
    
    

## Feature 16: Scan the Length of Top Level Domain

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1




## Prediction

def main(url):
    
    status = []
    
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))
    
    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))
    
    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))
    
    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url,fail_silently=True)
      
    status.append(tld_length(tld))
    
    return status




# Predict Function

def get_prediction_from_url(test_url):
    features_test = main(test_url)
    features_test = np.array(features_test).reshape((1, -1))
   
   # Predict result
    pred = model.predict(features_test) #unindent does not match any outer identation level (pyflakes E)

    if int(pred[0]) == 0:
        
        res="SAFE" 
        return res
    elif int(pred[0]) == 1.0:
        
        res="DEFACEMENT"
        return res
    elif int(pred[0]) == 2.0:
        res="PHISHING"
        return res
        
    elif int(pred[0]) == 3.0:
        
        res="MALWARE"
        return res




# Take user input and make prediction
url = input("Enter URL: ")
result = get_prediction_from_url(url)
print("Result:", result)
