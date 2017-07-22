# coding:utf-8

# python3

from bs4 import BeautifulSoup
import urllib.request as req
import os
from requests_oauthlib import OAuth1Session
import json

os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

### functions #############################################################################################
def init():
    return {
        'crawled_now' : '/home/vcp/stock_crawler/src/ch1/file.txt',
        'urls' : [
            'http://www.bengo4.com/topics/1000/',
        ],
        'selectors' : {
            'title' : 'title',
            'publish_date' : '.entry--open_date',
            'lawyer' : '#entryDetailProfileName a',
        },
        'labels' : {
            'url' : 'URL',
            'title' : 'タイトル',
            'publish_date' : '公開日時',
            'lawyer' : '取材協力弁護士名',
        },
    }

def crawl(url):
    res = req.urlopen(url)
    soup = BeautifulSoup(res, "html.parser")

    data = {}
    for k, v in settings['selectors']:
        data[k] = soup.find(v).string
    return data

def output(records):
    f = open(settings['crawled_now'], 'w')

    for url, record in records:
        for k, v in record:
            f.write(settings['labels'][k] + ":" + v + "\n")
    f.close()

### main #############################################################################################
settings = init()

records = []
for url in settings['urls']:
    records[url] = crawl(url)

output(records)