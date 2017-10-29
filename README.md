Parte 5: Reconocimiento de imágenes
===================



## Introducción
Cada imagen está compuesta de millones de puntos (píxeles). Cada píxel acumula información acerca del color, saturación y otros datos para formar la imagen visible.

Representación de un píxel:

    1 1 0 1 0
    0 1 0 1 1
    0 0 0 0 1

Para trabajar con procesamiento de imágenes se utilizan **Redes neuronales convolucionales (CNNs)**.

## Tipos de imágenes
### RGB
Se representa con 3 valores posibles, uno para cada color (rojo, verde o azul). El rango de dichos valores es desde 0 a 255.

 - Rojo: 255, 0, 0
 - Verde: 0, 255, 0
 - Azul: 0, 0, 255

### Blanco y negro
Se representa con 1 único valor posible que almacena la información de intensidad. El rango de dicho valor es desde 0 a 1.

 - Rojo: 255, 0, 0
 - Verde: 0, 255, 0
 - Azul: 0, 0, 255

> **Nota:**
> TensorFlow permite representar ambos canales (blanco y negro o RGB). Cada imagen puede ser representada
mediante una matriz en 3 dimensiones.
> El número de canales depende del número de elementos necesarios, es decir:
> *Blanco y negro: (6, 6, 1)*
> - Tamaño de la matriz: 6x6
> - Número de elementos: 1 (intensidad de color)
> *RGB: (6, 6, 3)*
> - Tamaño de la matriz: 6x6
> - Número de elementos: 3 (rojo, verde, azul)
