from PIL import Image
import os

# Definir o diretório de origem e destino
src_dir = 'normal'
dst_dir = src_dir + '-resized'

# Criar o diretório de destino se ele não existir
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Obter uma lista de todos os arquivos no diretório de origem
files = os.listdir(src_dir)

# Definir o novo tamanho (largura, altura)
new_size = (500, 500)  # Altere para o tamanho desejado

# Iterar sobre todos os arquivos
for file in files:
    # Definir o caminho completo para o arquivo de origem
    src_file = os.path.join(src_dir, file)

    # Abrir a imagem
    img = Image.open(src_file)

    # Redimensionar a imagem
    img_resized = img.resize(new_size)

    # Converter a imagem para o modo 'RGB'
    img_resized = img_resized.convert('RGB')

    # Definir o caminho completo para o arquivo de destino
    dst_file = os.path.join(dst_dir, file)

    # Salvar a imagem redimensionada
    img_resized.save(dst_file)