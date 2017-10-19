# coding:utf-8
from bs4 import BeautifulSoup
import urllib.request as req
import os
from requests_oauthlib import OAuth1Session
import json

os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

### functions #############################################################################################
def init():
    path = os.path.abspath(os.path.dirname(__file__))
    return {
        'crawled_log' : path + '/press_released.log',
        'crawled_now' : path + '/press_release_now.log',
    }

def crawl_nikkei():
    url = "http://www.nikkei.com/pressrelease/?searchKeyword=%E4%B8%96%E7%95%8C%E5%88%9D%E3%80%80%E3%81%AB%E6%88%90%E5%8A%9F&au=0"
    res = req.urlopen(url)
    soup = BeautifulSoup(res, "html.parser")
    news_records = soup.select(".m-newsListDotBorder_link")

    f = open(settings['crawled_now'], 'w')
    for record in news_records:
        _str = record.find("span", {"class" : "m-newsListDotBorder_text"}).string
        if _str.count('に成功') or _str.count('を開発'):
            fbk = open(settings['crawled_log'], 'r')
            href = record.get("href")
            if fbk.read().count(href) < 1:
                print(href)
                f.write("http://www.nikkei.com" + href + "\n")
                fbk.close()
                fbk = open(settings['crawled_log'], 'a')
                fbk.write("http://www.nikkei.com" + href + "\n")
            fbk.close()
    f.close()

def tweet(tweet_list):
    twitter = OAuth1Session(
        's2pdn7qCF87cYcHaKPjvoNYl0',
        'PIWmcvN7Qlu76hnNI2ADRaYU3cclo139I0HRazl7uWeFm1XgsQ',
        '855057045992595456-TWXOYZcGabFanh9I8ATzkiVG7zFj4m2',
        'medD4a2rUHP0LlsAsuXqeZDZYjeowvHcp4bSpnJi0zkVL'
    )
    for line in tweet_list:
        txt = line.encode('utf-8')
        params = {"status": txt}
        req = twitter.post("https://api.twitter.com/1.1/statuses/update.json",params = params)

### main #############################################################################################
settings = init()

crawl_nikkei()
tweet(open(settings['crawled_now'], 'r'))
