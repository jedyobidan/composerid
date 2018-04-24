from bs4 import BeautifulSoup
import webbrowser
import urllib2
import urlparse
import requests
from time import sleep
import re
import wget
urlList = []
composers = ["Adam", "Alkan", "J.S. Bach", "Banchieri", "Beethoven", "Billings", "Bossi", "Brahms", "Buxtehude", "Byrd", "Chopin", "Clementi", "Corelli", "Dufay", "Dunstable", "Field", "Flecha", 
        "Foster", "Frescobaldi", "Gershwin", "Giovannelli", "Grieg", "Haydn", "Himmel", "Hummel", "Isaac", "Ives", "Joplin", "Josquin", "Landini", "Lassus", "Liszt", "MacDowell", "Mendelssohn",
        "Monteverdi", "Mozart", "Pachelbel", "Prokofiev", "Ravel", "Scarlatti", "Schubert", "Schumann", "Scriabin", "Sinding", "Sousa", "Turpin", "Scarlatti", "Vecchi", "Victoria", "Vivaldi", "Weber"]
def getAllUrl(url):
    try:
        page = urllib2.urlopen( url ).read()
    except:
        return []
    try:
        soup = BeautifulSoup(page, "html.parser")
        soup.prettify()
        for composer in composers:
            for anchor in soup.findAll('a', href = re.compile(composer)):
                urlList.append(anchor['href'])
    except urllib2.HTTPError, e:
        print(e)
def downloading_midi(urlList):
    for l in urlList:
        print(l)
        r = requests.get(l)
        soup = BeautifulSoup(r.content, 'lxml')
        for anchor in soup.findAll('a', href=True):
            if "midi" in anchor['href']:
                midi_link = anchor['href']
                try:
                    wget.download(midi_link, '/u/smaleki/workspace/nlp/final')
                except:
                    print("cant download")

if __name__ == "__main__":
    urls = getAllUrl('http://kern.ccarh.org/')
    getAllUrl(urls)
    downloading_midi(urlList)

