import cv2
import os

# Diretórios de entrada e saída
diretorio_entrada = "ROI-equalizado"
diretorio_saida = "ROI-denoised"

# Verificar se o diretório de saída existe, se não, criar
if not os.path.exists(diretorio_saida):
    os.makedirs(diretorio_saida)

# Iterar sobre cada arquivo no diretório de entrada
for nome_imagem in os.listdir(diretorio_entrada):
    # Construir o caminho completo para a imagem
    caminho_imagem = os.path.join(diretorio_entrada, nome_imagem)

    # Ler a imagem
    imagem = cv2.imread(caminho_imagem)

    # Aplicar a técnica de denoising
    imagem_denoised = cv2.fastNlMeansDenoisingColored(imagem, None, 10, 10, 7, 21)

    # Construir o caminho completo para a imagem de saída
    caminho_saida = os.path.join(diretorio_saida, nome_imagem)

    # Salvar a imagem denoised
    cv2.imwrite(caminho_saida, imagem_denoised)