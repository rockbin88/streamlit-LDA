import streamlit as st
import re
import numpy as np
import pandas as pd
from pprint import pprint
import PyPDF2

#Gensim

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Spacy for lemmatization
import spacy 

# Plotting 
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Stream lit 

# Page layout

st.set_page_config(page_title='Analyzing Federal Reserve committee minutes using Machine Learning', 
                   layout ='wide')

# Page title
st.title("Analyzing Fed reserve minutes")
st.header("Classification Edition")

#Side bar
st.sidebar.header("Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type='pdf')
st.sidebar.markdown(""" [Example PDF input file](URL)""")

#Main panel
st.subheader('Dataset')

def clean_text(data): 
    words = nltk.word_tokenize(data)
    
    # Create a list of all the punctuations we wish to remove
    punctuations = ['.', ',', '/', '!', '?', ';', ':', '(',')', '[',']', '-', '_', '%']
    
    # Remove all the special characters
    punctuations = re.sub(r'\W', ' ', str(data))
    
    # Initialize the stopwords variable, which is a list of words ('and', 'the', 'i', 'yourself', 'is') that do not hold much values as key words
    stop_words  = stopwords.words('english')
    newStopWords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'federal',
                   'reserve', 'session', 'january', 'february', 'march', 'april', 'may', 'june', 'july',
                   'august', 'september', 'october', 'november', 'december', 'also', 'month', 'year', 'would',
                    'provide', 'could', 'error', 'likely', 'keep', 'seek', 'new', 'use', 'expect', 'action', 
                    'percent', 'go', 'staff', 'note', 'quarter', 'decide', 'see', 'first', 'second', 'third', 
                    'next', 'might', 'need', 'since', 'member', 'participant', 'committee', 'agree', 'vote', 
                    'open', 'serve', 'manager', 'fully', 'allow', 'without', 'operate', 'holiday']
    stop_words.extend(newStopWords)
    
    # Getting rid of all the words that contain numbers in them
    w_num = re.sub(r'\w*\d\w*', '', data).strip()
    
    # remove all single characters
    data = re.sub(r'\s+[a-zA-Z]\s+', ' ', data)
    
    # Substituting multiple spaces with single space
    data = re.sub(r'\s+', ' ', data, flags=re.I)
    
    # Removing prefixed 'b'
    data = re.sub(r'^b\s+', '', data)
    
    # Removing non-english characters
    data = re.sub(r'^b\s+', '', data)
    
    # Return keywords which are not in stop words 
    keywords = [word for word in words if word not in stop_words or word not in punctuations or  word not in w_num]
    
    return keywords


# LDA Model 
@st.cache(suppress_st_warning=True)
def lda_model(corpus, id2word):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=6, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    
    vis = gensimvis.prepare(lda_model, corpus, id2word)
    
    return vis
    
#Read PDF

text = ''

if uploaded_file is not None: 
    
    pdf = PyPDF2.PdfFileReader(uploaded_file)
    
    number_of_pages = pdf.getNumPages()
    
    count = 2
    
    
    while count < number_of_pages: 
        page = pdf.getPage(count)
        count+= 1
        text += page.extractText().lower()
    
    
    #Clean PDF

    sp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
    clean_data = ' '.join(clean_text(text))
    filtered_data = sp(clean_data)

    text_output =[]
    for word in filtered_data:
        text_output.append(word.lemma_)

    doc = [simple_preprocess(key, deacc=True, min_len=3) for key in text_output]
    id2word = corpora.Dictionary(doc)
    corpus = [id2word.doc2bow(text, allow_update=True) for text in doc]
    
    vis = lda_model(corpus, id2word)
    
    
    html_string = pyLDAvis.prepared_data_to_html(vis)
    from streamlit import components
    components.v1.html(html_string, width=1300, height=800, scrolling=True)
        
    
else: 
    st.info("Awaiting for PDF file to be uploaded")
    if st.button('Press to use example PDF file'): 
        filename = open("https://github.com/rockbin88/streamlit-LDA/blob/main/fomcminutes20210127.pdf", 'rb')
        st.markdown('INPUT DOC')
        pdf = PyPDF2.PdfFileReader(uploaded_file)
    
        number_of_pages = pdf.getNumPages()
    
        count = 2
            
        while count < number_of_pages: 
            page = pdf.getPage(count)
            count+= 1
            text += page.extractText().lower()
            
    #Clean PDF

    sp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
    clean_data = ' '.join(clean_text(text))
    filtered_data = sp(clean_data)

    text_output =[]
    for word in filtered_data:
        text_output.append(word.lemma_)

    doc = [simple_preprocess(key, deacc=True, min_len=3) for key in text_output]
    id2word = corpora.Dictionary(doc)
    corpus = [id2word.doc2bow(text, allow_update=True) for text in doc]
    vis = lda_model(corpus, id2word)
    
    html_string = pyLDAvis.prepared_data_to_html(vis)
    from streamlit import components
    components.v1.html(html_string, width=1300, height=800, scrolling=True)

