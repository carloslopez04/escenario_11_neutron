# Importaciónde librerías
import streamlit as st 
from neuron import neurona
import numpy as np


st.image("img/neurona.jpg")
# Título de la aplicación
st.title('!Hola neurona pro¡')
st.write('Elige el número de entradas/pesos que tendrá la neurona')
# Slider actualizable
entradas = st.slider('Número de entradas/pesos', min_value=1, max_value=10, value=1)

# Pesos dinámicos
st.title('Peso')
peso = st.columns(entradas)
pesos = []
for i, col in enumerate(peso):
    peso = col.number_input(f'W{i+1}', key=f"peso_{i}")  # Clave única para pesos
    pesos.append(peso)
st.write(f"W = {pesos}")

# Datos de entrada dinámicos
st.title('Entrada')
entrada = st.columns(entradas)
valores = [] 
for o, colu in enumerate(entrada):
    valore = colu.number_input(f'X{o+1}', key=f"entrada_{o}")  # Clave única para entradas
    valores.append(valore)
st.write(f"X = {valores}")

# Columnas finales 
cf1, cf2 = st.columns(2)

with cf1:
    st.subheader('Sesgo')
    sesgo = st.number_input('Introduce el valor del sesgo')  # Clave única para sesgo

with cf2:
    st.subheader('Función de activación')
    x = st.selectbox('Elige una función de activación',["ReLu", "Sigmoid", "Tanh", "Binary_step"])  
    # Creación de la neurona
    n = neurona(peso=pesos, sesgo=sesgo, func=x)
    # Resultado de la neurona
    resultado = n.run(valores)

if st.button('Calcular'):
    st.write(f"Resultado: {resultado}")



# Footer de la aplicación
st.markdown('---')
st.markdown('Creado por: [Carlos López Muñoz](https://www.linkedin.com/in/carlos-l%C3%B3pez-mu%C3%B1oz8941/)')