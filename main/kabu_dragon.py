from selenium import webdriver
import sys, os, re, time
import urllib.request as req

MAIL = "katsudonwork@gmail.com"
PASS = "katsu3x100zy"

# PhantomJSのドライバーを得る --- (※1)
browser = webdriver.PhantomJS()
browser.implicitly_wait(3)

DOMAIN = "http://www.kabudragon.com"
DOWNLOAD_DIR = os.getcwd() + '/image/downloaded'

def download(url, path):
    try:
        print("[download]", url, path)
        req.urlretrieve(url, path)
        time.sleep(1) # --- 礼儀として1秒スリープ
    except Exception as e:
        print("[ERROR] failed to downlaod url=", url)

def pre():
    browser.get(DOMAIN + '/ranking/kai200.html')
    links = browser.find_elements_by_css_selector(".rankingFrame:nth-of-type(2) a")
    _links = []
    for a in links:
        link = {}
        link['href'] = a.get_attribute('href')
        print(link['href'])
        id = a.find_element_by_xpath('parent::node()/preceding-sibling::td[position()=1]')
        link['file_name'] = id.text + '.png'
        print(link['file_name'])
        _links.append(link)

    print(os.path.isdir(DOWNLOAD_DIR))
    if not os.path.isdir(DOWNLOAD_DIR) == True:
        os.makedirs(DOWNLOAD_DIR)

    for link in _links:
        browser.get(link['href'])
        img_href = browser.find_element_by_css_selector(".chart2").get_attribute('src')
        print(img_href)
        download(img_href, 'image/downloaded/' + link['file_name'])

pre()









