import bs4
import spacy
import requests
import time


class WikiPageRetriever:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_sm")

    def __call__(self, targets):
        print("Retrieving target pages.")

        out = []
        print("opening wiki pages source code...")
        for page in open(targets,'r'):
            try:
            #if 1==1:
                page = page.rstrip("\n")
                response = requests.get(page)
                html = response.content
                html_parsed = bs4.BeautifulSoup(html, 'html.parser')

                blob = {
                            'title': html_parsed.find('title').get_text(),                                 
                            'page_sentencized': self._sentencize(html_parsed.find('body').get_text()),
                            'page_one_piece': html_parsed.find('body').get_text(),
                            'url': page,
                            #'links' : [link.get('href') for link in html_parsed.find_all('a')]
                    }
                out.append(blob)
                time.sleep(2)
            except:
                print("Skipping Article...")
        return out

    def _sentencize(self, text):
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
