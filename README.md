PARTE 5 - Reconocimiento de imágenes (MNIST)
===================

 - [Parte 1.1 - Introducción](#parte-11---introducción)
 - [Parte 1.2 - MNIST](#parte-12---mnist)

## PARTE 1.1 - Introducción

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
Se representa con 1 único valor posible que almacena la información de intensidad (luminancia o brillo). El rango de dicho valor es desde 0 a 1.

-------------
## PARTE 1.2 - MNIST

Como podemos ver en la documentación de TensorFlow, cuando alguien aprende un lenguaje de programación nuevo suele aprender a ejecutar un Hello World. En aprendizaje automático el **Hello World** tiene su equivalencia a **MNIST**.

### 1.2.1 - ¿Qué es MNIST?

MNIST es un conjunto de datos para el uso en visión por computador. Se compone de una serie de dígitos manuscritos como los siguientes:
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples.png" alt="Example">
</p>
La base de datos de MNIST incluye también un conjunto de etiquetas para asociar cada dígito a una etiqueta correspondiente (0, 1, 2 ... 9). El objetivo principal del conjunto de datos es facilitar el reconocimiento de dígitos en imágenes.

### 1.2.2 - ¿Cuándo y para qué se utiliza?
Para el reconocimiento de imágenes generalmente se utiliza un modelo basado en Softmax Regression (uno de los más simples de la regresión logística multinomial. El modelo Softmax se basa en la regresión logística multinomial que se utiliza cuando la variable dependiente que queremos etiquetar es nominal.

Es decir, dicha variable puede etiquetarse en un conjunto de categorías que se excluyen entre sí y para los cuales hay más de dos categorías.

#### Ejemplo
Imaginemos que tenemos números escritos en papel y queremos desarrollar una aplicación móvil para reconocer dichos números. Cada uno de nosotros tenemos una forma diferente de escribir dichos dígitos y además, nunca los escribimos de la misma forma. Por ello, poder identificar los dígitos con lenguaje de programación regular seria muy difícil y nos inclinamos por un modelo basado en redes neuronales.

Como ya hemos visto en otras partes de los vídeos, un modelo basado en redes neuronales necesita una gran cantidad de ejemplos para enseñar (*train*) a nuestro programa a reconocer dichos dígitos. El papel principal de MNIST en este caso es proporcionarnos una base de datos de infinidad de formas de representar un mismo dígito, por así decirlo es como si tuviéramos la forma de representar los dígitos de miles de personas diferentes almacenados en una base de datos.

<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples2.jpg" alt="Example 2">
</p>

-------------
## PARTE 1.3 - Tutorial práctico MNIST
Los datos de la base de datos de MNIST están alojados en un CDN, el código fuente del ejemplo ([src/mnist.py](https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/src/mnist.py) contiene varias líneas para la descarga de los datos:


```python
from tensorflow.examples.tutorials.mnist import input_data

# Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```
