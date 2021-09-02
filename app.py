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
        submit_sel = st.form_submit_button(label=" Iniciar búsqueda ")
        if submit_sel:
            if droga:
                st.success("Ha iniciado la búsqueda de la droga '{}' en {}". format(droga, busqueda_sel))
            else:
                st.warning("Debe seleccionar una droga para realizar la búsqueda")

    st.write("")

def file_uploading():
    st.subheader("File uploading")
    st.write("")
    with st.form("frm_file_upload"):
        file_up = st.file_uploader("Seleccione un archivo (csv) para procesar: ", type='csv')
        submit_fil = st.form_submit_button(label=" Iniciar proceso ")
        if submit_fil:
            if file_up:
                st.success("Ha iniciado el proceso del archivo seleccionado")
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
    st.write("Franco Visitini")
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


if __name__ == '__main__':
    main()
