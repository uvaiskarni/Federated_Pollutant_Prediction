import os
import zipfile
from shutil import rmtree

import requests
import wget
from bs4 import BeautifulSoup

import utils


def download(years=range(2014, 2020)):
    st_dir = utils.RAW_DIR
    # os.system('mkdir -p ' + st_dir)
    if os.path.exists(st_dir):
        rmtree(st_dir)
    os.makedirs(st_dir)
    for year in years:
        base_url = "https://datavardluft.smhi.se/portal/concentrations-in-air?C=1&M=180&S=36244&S=34656&S=18643&S=159402&" + \
                   "S=34399&S=20415&S=37479&S=164905&S=157993&S=18644&S=159403&S=8780&S=18638&S=18639&S=157992&S=8779&S=18640&" + \
                   "S=18641&S=8781&S=36242&S=20416&P=391&P=10&P=8&P=9&P=7&P=5&P=6001&Y=%d&CU=1&AC=6&SC=1&SC=3&vs=0:47:0:0:0:0:0:0" % year
        html_doc = requests.get(base_url).text
        soup = BeautifulSoup(html_doc, 'html.parser')
        r = soup.find("a", text="Ladda ner alla")

        target = "https://datavardluft.smhi.se/portal/" + r["href"]
        tmp_dis = os.path.join(st_dir, 'tmp.zip')
        wget.download(target, out=tmp_dis)
        with zipfile.ZipFile(tmp_dis, "r") as zip_ref:
            zip_ref.extractall(os.path.join(st_dir, str(year)))

        os.remove(tmp_dis)


if __name__ == "__main__":
    download()
