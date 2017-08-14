# coding:utf-8
import urllib.request as req
import os

os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

start_year = 2007
end_year = 2017

### functions #############################################################################################
class NikkeiAveDownoader :

    def __init__(self):
        self.url = 'http://k-db.com/indices/I101/1d/{{year}}?download=csv'
        self.path = 'csv/indices_I101_1d_{{year}}.csv'

    def download(self, url):
        req.urlretrieve(
            self.url.replace('{{year}}', str(year)),
            self.path.replace('{{year}}', str(year)) )

### main #############################################################################################
if __name__ == "__main__":
    nikkeiAveDownoader = NikkeiAveDownoader()
    for year in range(start_year, end_year):
        nikkeiAveDownoader.download(year)
