import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import joblib
import lightgbm as lgb
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import eli5
from eli5.lime import TextExplainer

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
    st.write("")

def file_uploading():
    st.subheader("File uploading")
    st.write("")
    st.write("")

def sentiment_analysis():
    st.subheader("Sentiment Analysis")
    st.write("")
    st.write("")

def about():
    st.subheader("About")
    st.write("Integrantes:")
    st.write("Mariana Peinado")
    st.write("Juan Boirazian")
    st.write("Jorge Corro")
    st.write("Franco Visitini")
    st.write("Federico Vessuri")

if __name__ == '__main__':
    main()
