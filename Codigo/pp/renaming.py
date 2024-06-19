import os
import shutil

# Definir o diretório de origem e destino
src_dir = 'ROI'
dst_dir = 'ROINormalizado'

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
    dst_file = os.path.join(dst_dir, f'A-{i}.jpg')  # assumindo que as imagens são .jpg

    # Copiar o arquivo
    shutil.copy(src_file, dst_file)