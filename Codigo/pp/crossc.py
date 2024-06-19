import cv2
import os
import numpy as np

def detectar_padrao(imagem, padroes, limiar=0.5):
    melhor_correlacao = 0
    melhor_bbox = None

    # Converter a imagem para escala de cinza e para uint8
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    for padrao in padroes:
        correlacao = cv2.matchTemplate(imagem_gray, padrao, cv2.TM_CCOEFF_NORMED)
        _, max_correlacao, _, max_loc = cv2.minMaxLoc(correlacao)

        if max_correlacao > melhor_correlacao:
            melhor_correlacao = max_correlacao
            melhor_bbox = (*max_loc, padrao.shape[1], padrao.shape[0])  # Coordenadas (x, y, largura, altura)

    if melhor_correlacao >= limiar:
        return True, melhor_bbox
    else:
        return False, None


def processar_imagens(diretorio_entrada, diretorio_saida, padroes, limiar=0.7):
    for imagem_nome in os.listdir(diretorio_entrada):
        imagem_path = os.path.join(diretorio_entrada, imagem_nome)
        imagem = cv2.imread(imagem_path)

        # Detecção do padrão
        encontrado, area_interesse = detectar_padrao(imagem, padroes, limiar)

        if encontrado:
            # Recorte da área de interesse
            x, y, w, h = area_interesse
            area_recortada = imagem[y:y + h, x:x + w]

            # Salvar a área recortada
            cv2.imwrite(os.path.join(diretorio_saida, imagem_nome), area_recortada)
            print(f'Padrão detectado em {imagem_nome}. Área de interesse recortada e salva.')
        else:
            print(f'Padrão não detectado em {imagem_nome}.')


# Diretórios de entrada e saída
diretorio_entrada = "normal-equalizado"
diretorio_saida = "auto-correlacao-equalizado"

# Carregar padrões de área de interesse
padroes = []
diretorio_padroes = "ROI-equalizado"

for padrao_nome in os.listdir(diretorio_padroes):
    padrao_path = os.path.join(diretorio_padroes, padrao_nome)
    padrao = cv2.imread(padrao_path, cv2.IMREAD_GRAYSCALE)  # Carregar em escala de cinza

    # Converter o tipo de dados do padrão para uint8
    padrao = padrao.astype(np.uint8)

    padroes.append(padrao)

# Processar imagens
processar_imagens(diretorio_entrada, diretorio_saida, padroes, limiar=0.8)