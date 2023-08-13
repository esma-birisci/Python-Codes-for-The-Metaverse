import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

df = pd.read_csv('C:/Users/Esma/Desktop/Kitap1.csv')

df = df[df['Battery'] == 'B0005']
df = df[df['Temperature_measured'] > 34] #choose battery B0005


##df['Time'] =pd.to_datetime(df['Time'], unit='s')
dfb = df.groupby(['id_cycle']).max()
dfb['Cumulated_T'] = dfb['Time'].cumsum()




import plotly.express as px
fig = px.scatter_matrix(dfb.drop(columns=['Time','type', 'ambient_temperature', 
                                          'time', 'Battery']), 
                                )
fig.update_traces(marker=dict(size=2,color='crimson',symbol='diamond')),
fig.update_traces(diagonal_visible=False)
fig.update_layout(
    title='Battery dataset',
    width=900,
    height=1200,
)
fig.update_layout({'plot_bgcolor': '#f2f8fd',
                   'paper_bgcolor': 'white',}, 
                    template='plotly_white',
                    font=dict(size=7)
                    )

fig.show()




fig = go.Figure()

fig.add_trace(go.Scatter(x=dfb['Cumulated_T']/3600, 
                         y=dfb['Capacity'],
                         mode='lines',
                         name='Capacity',
                         marker_size=3, 
                         line=dict(color='crimson', width=3)    
                        ))
fig.update_layout(
                  title="Battery discharge capacity",
                  xaxis_title="Working time [hours]",
                  yaxis_title=f"Battery capacity in Ahr"
    )
fig.update_layout({'plot_bgcolor': '#f2f8fd',
                   'paper_bgcolor': 'white',}, 
                    template='plotly_white')

fig.show()



from math import e
L = (dfb['Capacity']-dfb['Capacity'].iloc[0:1].values[0])/-dfb['Capacity'].iloc[0:1].values[0]
K = 0.13
L_1 = 1-e**(-K*dfb.index*dfb['Temperature_measured']/(dfb['Time']))
dfb['C. Capacity'] = -(L_1*dfb['Capacity'].iloc[0:1].values[0]) + dfb['Capacity'].iloc[0:1].values[0]
fig = go.Figure()

fig.add_trace(go.Scatter(x=dfb.index, 
                         y=dfb['C. Capacity'],
                         mode='lines',
                         name='Physical model',
                         line=dict(color='navy', 
                                   width=2.5,
                                   )))

fig.add_trace(go.Scatter(x=dfb.index, 
                         y=dfb['Capacity'],
                         mode='markers',
                         marker=dict(
                              size=4,
                              color='grey',
                              symbol='cross'
                                 ),
                         name='NASA dataset',
                         line_color='navy'))
fig.update_layout(
    title="Physical model comparison ",
    xaxis_title="Cycles",
    yaxis_title="ð¶, Capacity [Ahr]")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.9,
    xanchor="left",
    x=0.8
))

fig.update_layout({'plot_bgcolor': '#f2f8fd',
                  'paper_bgcolor': 'white',}, 
                   template='plotly_white')



M = pd.DataFrame()
S = pd.DataFrame()
def MAE(M,S):    
    return np.sum(S-M)/len(S)

print(f'Mean Absolute Error =', round(MAE(dfb['Capacity'], dfb['C. Capacity']), 3))



X_in = dfb['C. Capacity']          # input: the simulation time series
X_out = dfb['Capacity'] - dfb['C. Capacity']   # output: difference between measurement and simulation

X_in_train, X_in_test, X_out_train, X_out_test = train_test_split(X_in, X_out, test_size=0.33)
X_in_train.shape

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))



epochs = 100
loss = "mse"
model.compile(optimizer='adam',
              loss=loss,
              metrics=['mae'], #Mean Absolute Error
             )
history = model.fit(X_in_train, X_out_train, 
                    shuffle=True, 
                    epochs=epochs,
                    batch_size=20,
                    validation_data=(X_in_test, X_out_test), 
                    verbose=1)
fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(0, epochs, 1),
                         y=history.history['mae'],
                         mode='lines',
                         name=f'Training MAE',
                         marker_size=3, 
                         line_color='orange'))
fig.add_trace(go.Scatter(x=np.arange(0, epochs, 1),
                         y=history.history['val_mae'],
                         mode='lines',
                         name=f'Validation MAE',
                         line_color='grey'))

fig.update_layout(
                  title="Network training",
                  xaxis_title="Epochs",
                  yaxis_title=f"Mean Absolute Error")
fig.update_layout({'plot_bgcolor': '#f2f8fd' , 
                   'paper_bgcolor': 'white',}, 
                   template='plotly_white')

fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_in_train, 
                         y=X_out_train,
                         mode='markers',
                         name=f'Modelled Capacity',
                         marker=dict(
                              size=4,
                              color='grey',
                              symbol='cross'
                                 ), 
                        line_color='crimson'))
fig.add_trace(go.Scatter(x = X_in_train, 
                         y=model.predict(X_in_train).reshape(-1),
                         mode='lines',
                         name=f'Trained Capacity',
                         line=dict(color='navy', width=3)))
fig.update_layout(
    title="Network training",
    xaxis_title="Modelled capacity",
    yaxis_title="Î” (Mod. Capacity - Measured Cap.)")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.45
))
fig.update_layout({'plot_bgcolor': '#f2f8fd' , #or azure
'paper_bgcolor': 'white',}, template='plotly_white')








X_twin = X_in + model.predict(X_in).reshape(-1)

fig = go.Figure()

fig.add_trace(go.Scatter(x=dfb.index, 
                         y=X_twin,
                         mode='lines',
                         name=f'Hybrid digial twin',
                         line=dict(color='firebrick', width=3)))
fig.add_trace(go.Scatter(x=dfb.index, 
                         y=dfb['C. Capacity'],
                         mode='lines',
                         name=f'Modelled capacity',
                         line=dict(color='navy', 
                                   width=3,
                                   dash='dash')))

fig.add_trace(go.Scatter(x=dfb.index, 
                         y=dfb['Capacity'],
                         mode='markers',
                         marker=dict(
                              size=4,
                              color='grey',
                              symbol='cross'
                                 ),
                         name=f'Observed capacity',
                         line_color='navy'))
fig.update_layout(
    title="Comparison of hybrid twin with other models",
    xaxis_title="Cycles",
    yaxis_title="Capacity in Ahr")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.77
))
fig.update_layout({'plot_bgcolor': '#f2f8fd',
                  'paper_bgcolor': 'white',}, 
                   template='plotly_white')



fig.show()


cycles = np.arange(168,500,1)
temperature = dfb['Temperature_measured'].iloc[167]
time = dfb['Time'].iloc[167]
K = 0.13
L_e = 1-e**(-K*cycles*temperature/time)
X_in_e = -(L_e*dfb['Capacity'].iloc[0:1].values[0]) + dfb['Capacity'].iloc[0:1].values[0]
C_twin_e = X_in_e + model.predict(X_in_e).reshape(-1)
X_twin = X_in + model.predict(X_in).reshape(-1)

fig = go.Figure()

fig.add_trace(go.Scatter(x=cycles, 
                         y=X_in_e,
                         mode='lines',
                         name=f'C modelled (predicted)',
                         line=dict(color='navy', 
                                   width=3,
                                   dash='dash')))
fig.add_trace(go.Scatter(x=cycles, 
                         y=C_twin_e,
                         mode='lines',
                         name=f'C Digital twin (predicted)',
                         line=dict(color='crimson', 
                                   width=3,
                                   dash='dash'
                                  )))

fig.add_trace(go.Scatter(x=dfb.index, 
                         y=X_twin,
                         mode='lines',
                         name=f'C Digital twin',
                         line=dict(color='crimson',
                                   width=2)))
fig.add_trace(go.Scatter(x=dfb.index, 
                         y=dfb['C. Capacity'],
                         mode='lines',
                         name=f'C modelled',
                         line=dict(color='navy', 
                                   width=2)))

fig.update_layout(
    title="Battery capacity prediction",
    xaxis_title="Cycles",
    yaxis_title="Battery capacity [Ahr]")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.72
))
fig.update_layout({'plot_bgcolor': '#f2f8fd',
                  'paper_bgcolor': 'white',}, 
                   template='plotly_white')


fig.show()
