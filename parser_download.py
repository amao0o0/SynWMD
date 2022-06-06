import stanza
import nltk
stanza.download(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
nltk.download('stopwords')
