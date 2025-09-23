# modelo_treino.py
import tensorflow as tf
from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

import numpy as np
import cv2
import os
import random

# --- CONFIGURAÇÕES ---
IMG_SIZE = (128, 128)
DATASET_PATH = "image"

# Função para carregar e processar imagens
def carregar_imagem(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)  # Escala de cinza
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normaliza
    return img

# Criar pares de imagens
def criar_pares():
    imagens = os.listdir(DATASET_PATH)
    pares = []
    labels = []
    
    for _ in range(100):  # 100 pares de treino
        if random.random() > 0.5:
            # Par positivo
            img_name = random.choice(imagens)
            img1 = carregar_imagem(os.path.join(DATASET_PATH, img_name))
            img2 = carregar_imagem(os.path.join(DATASET_PATH, img_name))
            label = 1
        else:
            # Par negativo
            img1 = carregar_imagem(os.path.join(DATASET_PATH, random.choice(imagens)))
            img2 = carregar_imagem(os.path.join(DATASET_PATH, random.choice(imagens)))
            label = 0
        
        pares.append([img1, img2])
        labels.append(label)
    
    return np.array(pares), np.array(labels)

# Criar dataset
pares, labels = criar_pares()
pares = pares.reshape((-1, 2, IMG_SIZE[0], IMG_SIZE[1], 1))

@register_keras_serializable()
def diferenca_absoluta(tensors):
    return tf.abs(tensors[0] - tensors[1])

# Modelo simples de comparação
def criar_modelo():
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)
    base_model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu')
    ])
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    feat_a = base_model(input_a)
    feat_b = base_model(input_b)
    
    # Distância absoluta entre vetores
    distancia = layers.Lambda(
    diferenca_absoluta,
    output_shape=(64,)
    )([feat_a, feat_b])
    
    output = layers.Dense(1, activation='sigmoid')(distancia)

    model = models.Model(inputs=[input_a, input_b], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = criar_modelo()

# Preparar dados para treino
X1 = pares[:,0]
X2 = pares[:,1]

# Treinar
model.fit([X1, X2], labels, epochs=10, batch_size=8)

# Salvar modelo
model.save("modelo_treinado.keras")
print("Modelo salvo!")
