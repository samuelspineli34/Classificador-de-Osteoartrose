import os
import shutil
from PIL import Image

# Definir o diretório de origem e destino
src_dir = 'ROINormalizado'
dst_dir = 'ROIEspelhado'

# Criar o diretório de destino se ele não existir
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Obter uma lista de todos os arquivos no diretório de origem
files = os.listdir(src_dir)

# Iterar sobre todos os arquivos
for i, file in enumerate(files, start=1):
    # Definir o caminho completo para o arquivo de origem
    src_file = os.path.join(src_dir, file)

    # Definir o caminho completo para o arquivo de destino
    dst_file = os.path.join(dst_dir, f'roi-{i}.jpg')  # assumindo que as imagens são .jpg

    # Copiar o arquivo
    shutil.copy(src_file, dst_file)

    # Abrir a imagem
    img = Image.open(dst_file)

    # Espelhar a imagem
    img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Salvar a imagem espelhada
    img_mirror.save(os.path.join(dst_dir, f'roi-{i+len(files)}.jpg'))