import cv2
import os

# Diretórios de entrada e saída
diretorio_entrada = "ROI-contrast"
diretorio_saida = "normal-equalizado"

# Verificar se o diretório de saída existe, se não, criar
if not os.path.exists(diretorio_saida):
    os.makedirs(diretorio_saida)

# Iterar sobre cada arquivo no diretório de entrada
for nome_imagem in os.listdir(diretorio_entrada):
    # Construir o caminho completo para a imagem
    caminho_imagem = os.path.join(diretorio_entrada, nome_imagem)

    # Ler a imagem em escala de cinza
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    # Equalizar o histograma da imagem
    imagem_equalizada = cv2.equalizeHist(imagem)

    # Construir o caminho completo para a imagem de saída
    caminho_saida = os.path.join(diretorio_saida, nome_imagem)

    # Salvar a imagem equalizada
    cv2.imwrite(caminho_saida, imagem_equalizada)