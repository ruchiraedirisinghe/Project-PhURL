#### PhURL - Web Based Phishing URL Detection & Learning Platform ####
#### Created By - Ruchira Edirisinghe ####
#### Plymouth Final Year Project - 2023 ####
#### file 2 ####




# Importing Data

import pandas as pd
import itertools
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from lightgbm import LGBMClassifier
import os
import seaborn as sns
from wordcloud import WordCloud




# Loading Datasets

df=pd.read_csv('malicious_phish.csv')

print(df.shape)
df.head()

df.type.value_counts()



##############################################################################################################################################################

# Plotting WordCloud

############ METHOD 1

#df_phish = df[df['type'] == 'Phishing']
#df_malware = df[df['type'] == 'Malware']
#df_deface = df[df['type'] == 'Defacement']
#df_benign = df[df['type'] == 'Benign']


#phish_url = " ".join(i for i in df_phish.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(phish_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#malware_url = " ".join(i for i in df_malware.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(malware_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#deface_url = " ".join(i for i in df_deface.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(deface_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#benign_url = " ".join(i for i in df_benign.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(benign_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

print(df['url'].isnull().sum())


############ METHOD 2

#df_phish = df[df['type'] == 'Phishing']
#df_malware = df[df['type'] == 'Malware']
#df_deface = df[df['type'] == 'Defacement']
#df_benign = df[df['type'] == 'Benign']

#phish_url = " ".join(i for i in df_phish.url if isinstance(i, str))
#phish_url = ''.join(e for e in phish_url if e.isalnum() or e.isspace())
#phish_url = phish_url.lower()
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(phish_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#malware_url = " ".join(i for i in df_malware.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(malware_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#deface_url = " ".join(i for i in df_deface.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(deface_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#benign_url = " ".join(i for i in df_benign.url)
#wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(benign_url)
#plt.figure( figsize=(12,14), facecolor='k')
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.tight_layout(pad=0)
#plt.show()

#if not df_phish.empty:
#    phish_url = " ".join(i for i in df_phish.url)
#    wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(phish_url)
#    plt.figure( figsize=(12,14), facecolor='k')
#    plt.imshow(wordcloud, interpolation='bilinear')
#    plt.axis("off")
#    plt.tight_layout(pad=0)
#    plt.show()


##############################################################################################################################################################




# Feature Engineering

import re

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
    
df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))

from urllib.parse import urlparse

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    
    # print match group or No matching pattern found by scanning them.
    
    if match:
        return 1
    else:
        return 0

df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))




## Feature 2: Scan whether the URL is a Google Link

from googlesearch import search

def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
df['google_index'] = df['url'].apply(lambda i: google_index(i))




# Scan how many dots the URL contains

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df['count.'] = df['url'].apply(lambda i: count_dot(i))




## Feature 3: Scan how many 'www's the URL contains

def count_www(url):
    url.count('www')
    return url.count('www')

df['count-www'] = df['url'].apply(lambda i: count_www(i))




## Feature 4: Scan how many '@'s the URL contains

def count_atrate(url):
     
    return url.count('@')

df['count@'] = df['url'].apply(lambda i: count_atrate(i))




## Feature 5: Scan how many ' / 's the URL contains

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))




## Feature 6: Scan how many embeds the URL contains

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))




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
    
df['short_url'] = df['url'].apply(lambda i: shortening_service(i))




## Feature 8: Scan the protocol the web URL is using

def count_https(url):
    return url.count('https')

df['count-https'] = df['url'].apply(lambda i : count_https(i))

def count_http(url):
    return url.count('http')

df['count-http'] = df['url'].apply(lambda i : count_http(i))




## Feature 9: Scan is there are any spaces contained in the URL

def count_per(url):
    return url.count('%')

df['count%'] = df['url'].apply(lambda i : count_per(i))

def count_ques(url):
    return url.count('?')

df['count?'] = df['url'].apply(lambda i: count_ques(i))

def count_hyphen(url):
    return url.count('-')

df['count-'] = df['url'].apply(lambda i: count_hyphen(i))

def count_equal(url):
    return url.count('=')

df['count='] = df['url'].apply(lambda i: count_equal(i))




## Feature 10: Scan the length of the URL

def url_length(url):
    return len(str(url))

df['url_length'] = df['url'].apply(lambda i: url_length(i))




## Feature 11: Scan the Hostname's length

def hostname_length(url):
    return len(urlparse(url).netloc)

df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))

df.head()




## Feature 12: Scan for suspicious words inside the URL

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))




## Feature 13: Scan the number of digits used

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


df['count-digits']= df['url'].apply(lambda i: digit_count(i))




## Feature 14: Scan the number of letters used

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

df['count-letters']= df['url'].apply(lambda i: letter_count(i))


################################################################################################################################################################

from urllib.parse import urlparse
from tld import get_tld
import os.path

################################################################################################################################################################


## Feature 15: Scan the First Directory Length

def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

df['fd_length'] = df['url'].apply(lambda i: fd_length(i))




## Feature 16: Scan the Length of Top Level Domain

df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))




df = df.drop("tld",1)
df.columns
df['type'].value_counts()




##############################################################################################################################################################

# Exploratory Data Analysis


## Figure Generate 1:  Distribution of the use of IP

import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=df, hue="use_of_ip")


## Figure Generate 2: Distribution of Abnormal URL

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=df, hue="abnormal_url")


## Figure Generate 3: Distribution of Google Index

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=df, hue="google_index")


## Figure Generate 4: Distribution of Shortened URLs

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=df, hue="short_url")


## Figure Generate 5: Distribution of Suspicious URLs

sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=df, hue="sus_url")


## Figure Generate 6: Distribution of count of [.] dot

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count.", kind="box", data=df)


## Figure Generate 7: Distribution of count of WWWs

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count-www", kind="box", data=df)


## Figure Generate 8: Distribution of the count of "@"s

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count@", kind="box", data=df)


## Figure Generate 9: Distribution of the count of "dir"s

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="count_dir", kind="box", data=df)

## Figure Generate 10: Distribution of the host name length

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="hostname_length", kind="box", data=df)


## Figure Generate 11: Distribution of the first directory length

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="fd_length", kind="box", data=df)


## Figure Generate 9: Distribution of the top level domain length

sns.set(style="darkgrid")
ax = sns.catplot(x="type", y="tld_length", kind="box", data=df)

##############################################################################################################################################################




# Label Encording

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])




#Predictor Variables

# filtering out google_index as it has only 1 value

X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable

y = df['type_code']




# Training & Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)



################################################################################################################################################################

## Model Building