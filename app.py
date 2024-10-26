import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import os
import plotly.express as px
import keras

#MODELO TRENDS DE GOOGLE

length = 1
# Cargar el modelo y el escalador
model = load_model("modelo_lstm.keras")
model_exp = load_model("modelo_exportaciones.keras")
model_imp = load_model("modelo_importaciones.keras")
scaler = MinMaxScaler(feature_range=(0, 1))

# Configuración de la página de Streamlit
st.set_page_config(page_title="Analytics", page_icon=":bar_chart:",layout="wide")
st.title(":chart_with_upwards_trend: Squad Tech Analytics")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.write("Esta aplicación predice el siguiente valor de una serie de tiempo usando un modelo LSTM.")

# Cargar los datos
data = pd.read_csv("trendPapaya.csv",usecols=[1])
dataset = data.values
dataset = dataset.astype('float32') 

# Escalar los datos y preparar las entradas
dataset = scaler.fit_transform(dataset)

# Seleccionar los últimos 60 datos para hacer la predicción
train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
length =  1
batch_size = 10
train_generator = TimeseriesGenerator(train,train,length=length,batch_size=batch_size)
validation_generator = TimeseriesGenerator(test, test, length=length ,batch_size=batch_size)
# Hacer la predicción y desescalar
trainPredict = model.predict(train_generator)
testPredict = model.predict(validation_generator)
valdPredict = model.predict(testPredict)

trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverse = scaler.inverse_transform(train)
testPredict = scaler.inverse_transform(testPredict)
testY_inverse = scaler.inverse_transform(test)
valdPredict = scaler.inverse_transform(valdPredict)

prediccion = valdPredict
# Mostrar la predicción en la aplicación
st.subheader("Predicción del siguiente valor:")


# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(seq_size*2)-1:len(dataset)-1, :] = testPredict
testPredictPlot[len(train)+(length)-1:len(dataset)-1, :] = testPredict


trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[length:len(trainPredict)+length, :] = trainPredict

valPredictPlot = np.empty_like(dataset)
valPredictPlot.resize((len(dataset)+90,1))
valPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(seq_size*2)-1:len(dataset)-1, :] = testPredict
valPredictPlot[len(dataset)-1:len(dataset)+len(valdPredict)-1, :] = valdPredict


# Visualizar los datos históricos y la predicción
st.subheader("Datos Históricos y Predicción")


values2Show = valdPredict[0:13]

fig = px.line( values2Show, color_discrete_sequence=["#0514C0"]).update_layout(xaxis_title="Semanas", yaxis_title="Cambio en trending de Google")
fig.update_layout( plot_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)


#MODELO EXPORTACIONES

df = pd.read_csv('exp2024alimentos.csv')

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 2

past = 20
future = 6
learning_rate = 0.001
batch_size = 10
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

months = df['mes']
df = df.drop('mes', axis = 1)
df = normalize(df.values, train_split)
df = pd.DataFrame(df)
df.head()

train_data = df.loc[0 : train_split - 1]
val_data = df.loc[train_split:]


start = past + future
end = start + train_split

x_train = train_data[[i for i in range(12)]].values
y_train = df.iloc[start:end][[1]]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(12)]].values
y_val = df.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

exp2Show = model_exp.predict(dataset_val)

exp2Show = exp2Show[0:6]

fig = px.line( exp2Show, color_discrete_sequence=["#0514C0"]).update_layout(xaxis_title="meses", yaxis_title="Cambio en exportaciones")
fig.update_layout( plot_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

#MODELO IMPORTACIONES

df = pd.read_csv('imp2024alimentos.csv')

months = df['months']
df = df.drop('months', axis = 1)
df = normalize(df.values, train_split)
df = pd.DataFrame(df)
df.head()

train_data = df.loc[0 : train_split - 1]
val_data = df.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(5)]].values
y_train = df.iloc[start:end][[1]]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(5)]].values
y_val = df.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

imp2Show = model_imp.predict(dataset_val)

imp2Show = imp2Show[0:6]

fig = px.line( imp2Show, color_discrete_sequence=["#0514C0"]).update_layout(xaxis_title="meses", yaxis_title="Cambio en importaciones")
fig.update_layout( plot_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)