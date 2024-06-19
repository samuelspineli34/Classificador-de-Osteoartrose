import cv2
import os
import numpy as np

def processar_imagens(diretorio_entrada, diretorio_saida, padroes):
    # Inicializar o detector/descritor ORB
    orb = cv2.ORB_create()

    # Inicializar o correspondente de características
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for imagem_nome in os.listdir(diretorio_entrada):
        imagem_path = os.path.join(diretorio_entrada, imagem_nome)
        imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

        # Detectar e descrever características na imagem
        kp_imagem, desc_imagem = orb.detectAndCompute(imagem, None)

        if desc_imagem is None:
            print(f'Nenhum descritor encontrado em {imagem_nome}.')
            continue

        melhor_correspondencia = None
        melhor_bbox = None

        for padrao, kp_padrao, desc_padrao in padroes:
            if desc_padrao is None:
                print(f'Nenhum descritor encontrado no padrão.')
                continue

            # Encontrar correspondências entre a imagem e o padrão
            correspondencias = bf.match(desc_imagem, desc_padrao)

            # Ordenar as correspondências pela distância (quanto menor a distância, melhor a correspondência)
            correspondencias = sorted(correspondencias, key=lambda x: x.distance)

            # Considerar uma correspondência como boa se a distância for menor do que um certo valor
            if correspondencias and correspondencias[0].distance < 70:
                melhor_correspondencia = correspondencias[0]
                pt = np.array([kp_imagem[melhor_correspondencia.queryIdx].pt], dtype=np.int32)
                melhor_bbox = cv2.boundingRect(pt)

        if melhor_bbox:
            x, y, w, h = melhor_bbox
            area_recortada = imagem[y:y + h, x:x + w]

            cv2.imwrite(os.path.join(diretorio_saida, imagem_nome), area_recortada)
            print(f'Padrão detectado em {imagem_nome}. Área de interesse recortada e salva.')
        else:
            print(f'Padrão não detectado em {imagem_nome}.')

# Diretórios de entrada e saída
diretorio_entrada = "normal"
diretorio_saida = "orb"

# Carregar padrões de área de interesse
padroes = []
diretorio_padroes = "ROIEspelhado"

# Inicializar o detector/descritor ORB
orb = cv2.ORB_create()

for padrao_nome in os.listdir(diretorio_padroes):
    padrao_path = os.path.join(diretorio_padroes, padrao_nome)
    padrao = cv2.imread(padrao_path, cv2.IMREAD_GRAYSCALE)

    # Detectar e descrever características no padrão
    kp_padrao, desc_padrao = orb.detectAndCompute(padrao, None)

    if desc_padrao is not None:
        padroes.append((padrao, kp_padrao, desc_padrao))

# Processar imagens
processar_imagens(diretorio_entrada, diretorio_saida, padroes)