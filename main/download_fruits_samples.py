# coding:utf-8
from bs4 import BeautifulSoup
import urllib.request as req
import os
from requests_oauthlib import OAuth1Session
import json

os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

### functions #############################################################################################
def init():
    return {
        'crawled_log' : '/home/vcp/stock_crawler/src/ch1/file_bk.txt',
        'crawled_now' : '/home/vcp/stock_crawler/src/ch1/file.txt',
    }

def getSoup(url, soup=BeautifulSoup(res, "html.parser")):
    res = req.urlopen(url)
    return soup

def download(url):
    req.urlretrieve(url, os.path.basename(url))

def crawl():
    soup = getSoup("https://www.google.co.jp/imghp?hl=ja&tab=wi&authuser=0")
    soup = getSoup("https://www.google.co.jp/search?q=%E3%82%8A%E3%82%93%E3%81%94&safe=off&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjz857EqZLSAhVFk5QKHcp2DPMQ_AUICCgB&biw=1280&bih=703")
    imgs = soup.findAll("img", {"class":"rg_ic"})
    for img in imgs:
        download(img['src'])


### main #############################################################################################
settings = init()
crawl()
