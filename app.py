import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import joblib
import lightgbm as lgb
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import eli5
from eli5.lime import TextExplainer

import praw
import re
from datetime import datetime
import twint
import nest_asyncio
# nest_asyncio.apply()

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def main():
    #   Con esto se configura un titulo e icono para la página web
    #   st.set_page_config(page_title="Grupo 6 - TP 4 App", page_icon="icon_g6_tp4.png", initial_sidebar_state="auto")
    st.title('TP 4 - App Grupo 6')
    st.header("Digital House - Data Science - TP 4 - App Grupo 6")

    opciones = ["Text scrapping", "File uploading", "Sentiment Analysis", "About"]
    opcion_sel = st.sidebar.selectbox("Option Menu", opciones)

    if opcion_sel == "Text scrapping":
        text_scrapping()

    if opcion_sel == "File uploading":
        file_uploading()

    if opcion_sel == "Sentiment Analysis":
        sentiment_analysis()

    if opcion_sel == "About":
        about()


def text_scrapping():
    st.subheader("Text scrapping")
    st.write("")
    with st.form("frm_text_scrap"):
        droga = st.text_input("Nombre de la droga: ")
        busqueda = ["Twitter", "Reddit"]
        busqueda_sel = st.selectbox("Opciones de búsqueda", busqueda)
        subreddit = st.text_input("Nombre del subreddit: ")
        submit_sel = st.form_submit_button(label=" Iniciar búsqueda ")
        if submit_sel:
            if droga:
                if busqueda_sel == "Twitter":
                    st.success("Ha iniciado la búsqueda de la droga '{}' en {}". format(droga, busqueda_sel))
                else:
                    if subreddit:
                        st.success("Ha iniciado la búsqueda de la droga '{}' en {}". format(droga, busqueda_sel))
                    else:
                        st.warning("Debe seleccionar un subreddit para realizar la búsqueda")
            else:
                st.warning("Debe seleccionar una droga para realizar la búsqueda")

    st.write("")

def file_uploading():
    st.subheader("File uploading")
    st.write("")
    with st.form("frm_file_upload"):
        st.write("El archivo debe ser formato csv y debe contener al menos las siguientes columnas: date, review, droga")
        file_upr = st.file_uploader("Seleccione un archivo (csv) para procesar: ", type='csv')
        submit_fil = st.form_submit_button(label=" Iniciar proceso ")
        if submit_fil:
            if file_upr:
                st.success("Ha iniciado el proceso del archivo seleccionado")
                procesar_archivo(file_upr)
            else:
                st.warning("Debe seleccionar un archivo antes de iniciar el proceso")
    st.write("")

def sentiment_analysis():
    st.subheader("Sentiment Analysis")
    st.write("")
    with st.form("frm_file_upload"):
        text_val = st.text_area("Ingrese el texto para analizar: ")
        submit_txt = st.form_submit_button(label=" Iniciar análisis de sentimiento ")
        if submit_txt:
            if text_val:
                st.success("Ha iniciado el análisis del texto ingresado")
            else:
                st.warning("Debe ingresar un texto antes de iniciar el análisis")
    st.write("")

def about():
    st.subheader("About")
    st.write("Integrantes:")
    st.write("Mariana Peinado")
    st.write("Juan Boirazian")
    st.write("Jorge Corro")
    st.write("Franco Visintini")
    st.write("Federico Vessuri")

def buscar_tweets(droga):
    #### BUSCO TWEETS QUE CONTENGAN LA FRASE "Droga is" y despues lo guardo en el DF Tweets_df
    c = twint.Config()
    Busqueda =   """\"""" + droga + " is" + """\""""
    c.Search = Busqueda
    c.Limit = 200
    c.Pandas = True
    c.Lang="en"
    twint.run.Search(c)
    Tweets_df = twint.storage.panda.Tweets_df
    Tweets_df['droga'] = droga
    Tweets_df=Tweets_df[['date','tweet','droga']]
    Tweets_df.rename(columns={'tweet': 'review'}, inplace=True)
    c.Hide_output = True
    return Tweets_df

def buscar_reddit(subredd , droga):
    i=0
    column_names = ["droga", "review", "date"]
    df = pd.DataFrame(columns = column_names)
    Subreddit = subredd ### el subreddit donde quiero hacer la busqueda
    Busqueda =   """\"""" + droga + " is" + """\""""
    reddit = get_reddit_credentials()
    subR = reddit.subreddit(Subreddit)
    resp = subR.search(Busqueda,limit=100)
    for submission in resp:
        df.at[i, 'droga'] = droga
        df.at[i, 'review'] =str(str(submission.title.encode('ascii', 'ignore').decode("utf-8")) +" "+ str(submission.selftext[:120].encode('ascii', 'ignore').decode("utf-8")))        
        df.at[i, 'date'] = datetime.utcfromtimestamp(int((submission.created_utc))).strftime('%Y-%m')
        i+=1
    return df

def get_reddit_credentials():
    return praw.Reddit(client_id='5U6IG9mVmOBz08m7gb_z8Q',client_secret='Y8yZhKAmDk6ryyEiXutrM0SVgnAMEg',username='jboirazian',password='+xj<_6$9hsZ7E)L',user_agent='jboirazian_grupo4')

def limpiar_texto(texto):
    # poner todo en minúscula
    minuscula = texto.lower()   
    # quitar patrón de repetición
    pattern=minuscula.replace('&#039;', "'")
    # Remover caracteres especiales
    special_remove = pattern.replace(r'[^\w\d\s]',' ')    
    # Remover los caracteres no ASCII
    ascii_remove = special_remove.replace(r'[^\x00-\x7F]+',' ')    
    # Remover espacios en blanco iniciales y finales
    whitespace_remove = ascii_remove.replace(r'^\s+|\s+?$','')    
    # Replazar multiples espacios en blanco con un ùnico espacio
    multiw_remove = whitespace_remove.replace(r'\s+',' ')    
    # Replazar 2 o más puntos por 1 solo punto
    dots_remove = multiw_remove.replace(r'\.{2,}', ' ')    
    return dots_remove

def contar_palabras(s):
    return (len(s.split()))

# Esta función aplica un tokenizador, genera stems y borra stop_words
def clean_datos(review_text, tokenizer, stemmer, stopwords):    
    stopwords_stem = [stemmer.stem(x) for x in stopwords]    #tokens (eliminamos todos los signos de puntuación)
    words = tokenizer.tokenize(review_text)
    # stemming: raiz y minúsculas:
    stem_words = [stemmer.stem(x) for x in words]
    # eliminamos stopwords (ya pasaron por stem)
    clean_words = [x for x in stem_words if x not in stopwords_stem]
    result = " ".join(clean_words)
    return(result)

def procesar_archivo(arch):
    if arch is not None:
        df = pd.read_csv(arch)
        if df is not None:
            st.write("dataframe del archivo:", df.head(5))
            st.write("shape: ", df.shape)
            procesar_dataframe(df)
        else:
            st.warning("No hay datos en el archivo para procesar")
    else:
        return None

def procesar_dataframe(df):
    # Se limpian las reviews del dataframe
    st.write("Se limpian las reviews del dataframe...")
    df.review.apply(limpiar_texto)
    # Definimos tokenizador, stemmer y stop_words que utilizaremos en la función "clean_datos"
    tokenizer = RegexpTokenizer(r"\w+")
    englishStemmer = SnowballStemmer("english")
    stopwords_en = stopwords.words('english')
    # Se quitan las stopwords y se stemizan las palabras limpias de las reviews del dataframe
    st.write("Se quitan las stopwords y se stemizan las palabras limpias de las reviews del dataframe...")
    clean_review = [clean_datos(x, tokenizer, englishStemmer, stopwords_en) for x in df.review]
    st.write("clean_review: ", clean_review)
    # se vectorizan las reviews del dataframe
    st.write("se vectorizan las reviews del dataframe...")
    count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords_en)
    # se genera la matriz esparsa
    st.write("se genera la matriz esparsa...")
    clean_review_sparse = count_vectorizer.transform(clean_review)
    # se le aplica una transformación con tfidf
    st.write("se le aplica una transformación con tfidf...")
    tfidf_transformer = TfidfTransformer(sublinear_tf=True,  norm='l2')
    clean_review_tfidf = tfidf_transformer.transform(clean_review_sparse)
    # se le aplica una transfomación con SVD
    st.write("se le aplica una transfomación con SVD...")
    svd = TruncatedSVD(n_components = 200)
    clean_review_svd = svd.transform(clean_review_tfidf)

if __name__ == '__main__':
    main()
