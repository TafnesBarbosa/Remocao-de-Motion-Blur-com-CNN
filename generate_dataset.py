import cv2
import numpy as np
import tensorflow as tf
import os
import random as rd

def resize(input_image):
    """
    Função que muda o tamanho da imagem de entrada 'input_image' para 128x128.
    """

    # Muda o tamanho da imagem atravpes do método nearest
    input_image = tf.image.resize(input_image, (128, 128), method="nearest")

    # Transforma a imagem em um array
    input_image = np.array(input_image)
    return input_image

def apply_motion_blur(input_image, random=True, ks=5, horizontal=True):
    """
    Função que aplica o motion blur, horizontal ou vertical, de forma aleatória ou não a uma imagem 'input_image'.

    - random diz se a escolha do tamanho do kernel vai ser aleatoria.
    - ks é usado quando random=False e é o tamanho do kernel a ser usado.
    - horizontal=True diz se o motion blur vai ser horizontal e quando for False, o motion blur é vertical.
    """

    # Escolhe o tamanho do kernel de forma aleatoria e uniforme, ou usa-se o tamanho fornecido como argumento da função
    if random:
        kernel_size = rd.choice([3,5,7,9,11,13])
    else:
        kernel_size = ks

    # Criação do kernel.
    ## Preenche a linha do meio, se horizontal, ou a coluna do meio, se vertical, com uns.
    kernel = np.zeros((kernel_size, kernel_size))
    if horizontal:
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    else:
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    ## Normalização (divide cada termo pelo tamanho do kernel, para que a soma dos termos do kernel seja igual a 1)
    kernel /= kernel_size
      
    # Aplica-se o kernel à imagem de entrada.
    output_image = cv2.filter2D(input_image, -1, kernel)
      
    return output_image, kernel_size

def data_load_and_generate():
    """
    Função que carrega as imagens do dataset original, 
    aplica um resize nelas para que fiquem em 128x128, 
    aplica o motion blur horizontal e salva as imagens
    sem distorção e com distorção nas pastas 'images' 
    e 'images_blurred', respectivamente.

    Esse arquivo deve estar na mesma pasta que a pasta com as imagens originais.
    A pasta original deve se chamar 'images_orig'.
    """

    # Toma-se o número de imagens do dataset
    num_images = len(os.listdir("images_orig"))

    i = 1
    for file in os.listdir("images_orig"):
        # Carrega a imagem original
        image = cv2.imread("images_orig/" + file)

        # Muda o tamanho da imagem para 128x128
        image = resize(image)

        # Aplica o motion blur horizontal na imagem
        output_image, kernel_size = apply_motion_blur(image, horizontal=True)

        # Salva a imagem com motion blur
        cv2.imwrite("images_blurred/" + str(kernel_size) + "_" + file[:-4] + "_blurred.jpg", output_image)

        # Salva a imagem sem motion blur, mas com tamanho 128x128
        cv2.imwrite("images/" + file, image)

        # Mostra o andamento do procedimento
        print(round(i / num_images * 100, 2), "%")
        i += 1

# Chama a função principal
data_load_and_generate()