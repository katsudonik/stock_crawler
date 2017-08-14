# coding:utf-8
import urllib.request as req
import os
import sys

os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

start_year = 2007
end_year = 2017

try:
    relative_url = sys.argv[1]
except IndexError:
    relative_url = 'indices/I101/1d'

### functions #############################################################################################
class NikkeiAveDownoader :

    def __init__(self):
        self.url = 'http://k-db.com/' + relative_url + '/{{year}}?download=csv'
        self.path = os.path.dirname(os.path.abspath(__file__)) + '/csv/' + relative_url.replace('/', '_') + '_{{year}}.csv'

    def download(self, url):
        req.urlretrieve(
            self.url.replace('{{year}}', str(year)),
            self.path.replace('{{year}}', str(year)) )

### main #############################################################################################
if __name__ == "__main__":
    nikkeiAveDownoader = NikkeiAveDownoader()
    for year in range(start_year, end_year):
        nikkeiAveDownoader.download(year)
