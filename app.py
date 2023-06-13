import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential(
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.initialize()

iris = sns.load_dataset('iris')

st.title('Iris EDA')
line_count = st.slider('Select a line count', 1, 10, 3)
st.write(f'You selected {line_count} lines.')
st.dataframe(iris)