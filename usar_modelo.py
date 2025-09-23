# usar_modelo.py
import tensorflow as tf
import cv2
import tensorflow as tf
from modelo_treino import diferenca_absoluta

IMG_SIZE = (128, 128)

model = tf.keras.models.load_model(
    "modelo_treinado.keras",
    custom_objects={"diferenca_absoluta": diferenca_absoluta},
    compile=False
)
print("Modelo carregado com sucesso!")

def carregar_imagem(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

# Carregar modelo
model = tf.keras.models.load_model("modelo_treinado.keras", compile = False)

# Testar com duas imagens
img1 = carregar_imagem("image/teste.png")
img2 = carregar_imagem("image/test.png")

pred = model.predict([img1, img2])
print("Similaridade:", pred[0][0])
