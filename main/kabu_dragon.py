from selenium import webdriver
import sys, os, re, time
import urllib.request as req

MAIL = "katsudonwork@gmail.com"
PASS = "katsu3x100zy"

# PhantomJSのドライバーを得る --- (※1)
browser = webdriver.PhantomJS()
browser.implicitly_wait(3)

DOMAIN = "http://www.kabudragon.com"

def download(url, path):
    try:
        print("[download]", url, path)
        req.urlretrieve(url, path)
        time.sleep(1) # --- 礼儀として1秒スリープ
    except Exception as e:
        print("[ERROR] failed to downlaod url=", url)

def pre():
    browser.get(DOMAIN + '/ranking/kai200.html')
    links = browser.find_elements_by_css_selector("rankingFrame > a")
    for a in links:
        href = a.get_attribute('href')
        print(href)
        browser.get(DOMAIN + href)
        img_href = browser.find_element_by_css_selector(".chart2").get_attribute('src')
        print(img_href)
        download(img_href, 'img.jpg')

pre()










