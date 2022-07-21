https://raw.githubusercontent.com/fridaruh/dashboard_streamlit/master/income.csv

import streamlit as st
import pandas as pd

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

siteHeader = st.beta_container()
with siteHeader:
    st.title('Modelo de evaluación de ingresos')
    st.markdown(""" En este proyecto se busca encontrar cuáles son las características 
    principales que pueden predecir que una persona gane más o menos de $50 K anuales.""")

dataExploration = st.beta_container()
with dataExploration:
    st.header('Dataset: Ingresos')
    st.text("""Este dataset corresponde a una transformación del set de datos oficial proveniente del
    siguiente set de datos:""")


df = pd.read_csv('https://raw.githubusercontent.com/fridaruh/dashboard_streamlit/master/income.csv')

st.write(df.head())

st.subheader('Distrbuciones:')

distribution_sex = pd.DataFrame(df.sex.value_counts())

st.bar_chart(distribution_sex)

st.text('Con esta gráfica buscamos mostrar la distribución de los datos con respecto al sexo.')

newFeatures = st.beta_container()
with newFeatures:
    st.header('Nuevas variables: ')
    st.text('Demos un vistazo a las principales variables de este dataset: ')

st.markdown('* **first feature:** this is the explanation') 
st.markdown('* **second feature:** another explanation')

modelTraining = st.beta_container()
with modelTraining:
    st.header('Entrenamiento del modelo')
    st.text('En esta sección puedes hacer una selección de los hiperparámetros del modelo.')

df = df.drop(['Unnamed: 0','income','fnlwgt','capital-gain','capital-loss','native-country'], axis=1)
Y = df['income_bi']
df = df.drop(['income_bi'], axis=1)
X = pd.get_dummies(df, columns = ['race','sex','workclass','education','education-num','marital-status','occupation','relationship'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 15)

max_depth = st.slider ('¿Cuál debería ser el valor de max_depth para el modelo?', min_value=1, max_value=10, value=2, step=1)


t = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=max_depth)
model = t.fit(x_train, y_train)

prediction = model.predict(x_test)
score = model.score(x_train, y_train)

st.header('Performance del Modelo')
st.text('Score:') 
st.text(score)

st.markdown( """ <style>
 .main {
 background-color: #AF9EC;
}
</style>""", unsafe_allow_html=True )
