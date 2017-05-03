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
    browser.get(DOMAIN + '/%E6%A0%AA%E4%BE%A1/%E3%82%A2%E3%83%83%E3%83%97%E3%83%AB%E3%82%A4%E3%83%B3%E3%82%BF%E3%83%BC%E3%83%8A%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB')
    img_href = browser.find_element_by_css_selector(".chart2").get_attribute('src')
    download(img_href, 'img.jpg')

print(browser.current_url)
print(browser.page_source)
pre()










