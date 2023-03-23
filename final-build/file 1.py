'''

#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### Final Build - file 1 ####

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


# Define prediction function
def predict(url):
    
    # Extract features
    
    use_of_ip = having_ip_address(url)              #1
    abnormal_url = abnormal_url(url)                    #1
    google_index = google_index(url)                #2
    count_dot = count_dot(url)                          #2
    count_www = count_www(url)                      #3
    count_at = count_atrate(url)                    #4
    count_dir = no_of_dir(url)                      #5
    count_embed_domian = no_of_embed(url)           #6
    short_url = shortening_service(url)             #7
    count_https = count_https(url)                  #8
    count_http = count_http(url)                        #8
    count_per = count_per(url)                      #9
    count_ques = count_ques(url)                        #9
    count_hyphen = count_hyphen(url)                    #9
    count_equal = count_equal(url)                      #9
    url_length = url_length(url)                    #10
    hostname_length = hostname_length(url)          #11
    sus_url = suspicious_words(url)                 #12
    count_digits = digit_count(url)                 #13
    count_letters = letter_count(url)               #14
    fd_len = fd_length(url)                         #15
    tld = get_tld(url, fail_silently=True)          #15
    tld_len = tld_length(tld)                       #16
    

    # Create feature array
    features = np.array([use_of_ip, abnormal_url, google_index, count_dot, count_www,
                         count_at, count_dir, count_embed_domian, short_url, count_https, count_http, count_per,
                         count_ques, count_hyphen, count_equal, url_length, hostname_length, sus_url, count_digits,
                         count_letters, fd_len, tld_len, ]).reshape(1, -1)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Predict result
    pred = model.predict(features)
    
    # Map result to class label
    if int(pred[0]) == 0:
        return "SAFE"
    elif int(pred[0]) == 1:
        return "DEFACEMENT"
    elif int(pred[0]) == 2:
        return "PHISHING"
    elif int(pred[0]) == 3:
        return "MALWARE"

# Take user input and make prediction
url = input("Enter URL: ")
result = predict(url)
print("Result:", result)


# # Hardcoded example URL for testing
# url = 'towardsdatascience.com/random-forest-in-python-24d0893d51c0'

# # Make prediction for the example URL
# result = predict(url)

# # Print the result
# print("Result:", result)