import os
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = (224, 224)

#Carregar a imagem e separar
def carregar_imagem(caminho):
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Exemplo de modelo simples para teste
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extrair_vetor(img):
    return model.predict(img)

#Função para carregar todas as imagens do banco
def carregar_banco(pasta_banco):
    banco = {}
    for arquivo in os.listdir(pasta_banco):
        if arquivo.lower().endswith((".png", ".jpg", ".jpeg")):
            caminho = os.path.join(pasta_banco, arquivo)
            img = carregar_imagem(caminho)
            vetor = extrair_vetor(img)
            banco[arquivo] = vetor
    return banco

#Comparar nova imagem com o banco
def comparar_com_banco(nova_img_caminho, banco, limiar=0.9):
    nova_img = carregar_imagem(nova_img_caminho)
    vetor_novo = extrair_vetor(nova_img)

    for nome, vetor in banco.items():
        similaridade = cosine_similarity(vetor_novo, vetor)[0][0]
        if similaridade >= limiar:
            return f"Imagem encontrada: {nome} (similaridade {similaridade:.4f})"
    return "Nenhuma imagem igual encontrada"

#Exemplo de uso
if __name__ == "__main__":
    banco = carregar_banco("saida_pdf")
    resultado = comparar_com_banco("saida_pdf/pagina_1.jpg", banco, limiar=0.95)
    print (resultado)

#Simulação de imagem do banco
#img1 = carregar_imagem("image/test.png")
#img2 = carregar_imagem("image/tes.png")
#Nova imagem para comparação
#nova_img = carregar_imagem("image/teste.png")

#vetor1 = extrair_vetor(img1)
#vetor2 = extrair_vetor(img2)
#vetor_novo = extrair_vetor(nova_img)

#model.save("modelo_comparado.keras")

#Carrego o modelo
#model = tf.keras.models.load_model("modelo_comparado.keras", compile=False)

#model.save("modelo_comparado.keras")

#Comparar similaridade
#similaridade = cosine_similarity(vetor1, vetor2)[0][0]
#print("Similaridade entre doc1 e doc2: ", similaridade)


# Comparar com img1
#similaridade_com_img1 = cosine_similarity(vetor_novo, vetor1)[0][0]
#print("Similaridade com img1: ", similaridade_com_img1)

# Comparar com img2
#similaridade_com_img2 = cosine_similarity(vetor_novo, vetor2)[0][0]
#print("Similaridade com img2: ", similaridade_com_img2)
#'''*/