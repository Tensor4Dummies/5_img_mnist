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
## PARTE 1.3 - Tensores de MNIST

Los datos de la base de datos de MNIST están alojados en un CDN, dichos datos contienen:

 - 55.000 puntos de entrenamiento (*mnist.train*)
 - 10.000 puntos de test (*mnist.test*)
 - 5.000 puntos de validación (*mnist.validation*)

El motivo de tenerlos separados es porque si juntásemos los datos de entrenamiento con los de test o validación provocaría que se generalizase y la máquina no aprendería a etiquetar nuevos elementos ya que contaría con todos de por si.

En este caso práctico de MNIST vamos a emplear el código fuente del ejemplo ([src/mnist.py](https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/src/mnist.py)), contiene varias líneas para la descarga de los datos:


```python
from tensorflow.examples.tutorials.mnist import input_data

# Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
```

### 1.3.1 - Variables del modelo Softmax
El modelo Softmax trabaja con varias variables determinadas por el tipo de dato contenido en ellas por ello podemos definir varias:

 - X = Imagen
 - Y = Etiqueta

Ejemplo:
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples3.png" alt="Example 3">
</p>

En la imagen podemos observar una matriz de tamaño 28 x 28 en la que se almacenan datos respectivos a la intensidad de color de cada píxel como hemos explicado en el apartado 1.1 de introducción. En este caso, la variable X sería la imagen y la variable Y sería "1" al ser la etiqueta respectiva a dicha imagen.

#### Tensor que almacena los datos de las imágenes
Tanto el conjunto de entrenamiento como el de testing contienen las dos variables. El resultado de estos datos se compone de un **tensor** de la forma [55.000, 784], la primera dimensión será un índice de imágenes (la 55.000) y la segunda dimensión será un índice de cada píxel de cada imagen.

Es decir, cada elemento de nuestro **tensor** es un valor de intensidad (entre 0 y 1) para cada píxel de cada imagen particular, se puede entender mejor en la siguiente imagen:
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples4.png" alt="Example 4">
</p>

#### Tensor que almacena los datos de las etiquetas de cada imagen
Además, cada imagen de la base de datos MNIST contiene una etiqueta Y respectiva al valor correcto etiquetado sobre dicha imagen (valor de los dígitos de 0 a 9).

En nuestro ejemplo, representamos las etiquetas como vectores posicionales one-hot. En un vector one-hot se almacenan datos en el rango de valores de 0 a 1 y que cada dígito estará representado por "1" en la posición de dicho dígito. Es decir, para representar el dígito "5" tendremos un vector de la forma:

<p align="center"><i>[0, 0, 0, 0, 0, 5, 0, 0, 0, 0]</i></p>
Esto produce un **tensor** de la forma [55.000, 10], podemos entenderlo mejor en la siguiente imagen:
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples5.png" alt="Example 5">
</p>

-------------
## PARTE 1.4 - Funcionamiento del modelo Softmax
El modelo asigna probabilidades estadísticas de que una imagen pertenezca a un dígito, por lo que podemos encontrarnos ante la posibilidad de que el modelo asigne a una misma imagen varias probabilidades de pertenecer a cierto número. Para este problema es útil la regresión de softmax pues nos devuelve una lista de valores entre 0 y 1.

El modelo asigna primero evidencias a una imagen relativas a qué dígito corresponde, para ello realiza una **suma ponderada de las intensidades de los píxeles**. Dicha suma provoca que se denigren los valores que no coincidan entre la imagen y la etiqueta, es decir, **el peso de la suma ponderada será negativa si un pixel no se corresponde con la etiqueta y positiva en caso de que si lo sea.**

En el siguiente diagrama podemos entender mejor cómo son las sumas ponderadas para cada dígito. En azul se representa el peso positivo (se corresponde) y en rojo el peso negativo (no se corresponde).
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples6.png" alt="Example 6">
</p>

A estas sumadas ponderadas añadimos el **sesgo**, dicha cantidad es un peso fijo de la red neuronal para compensar que ciertas entradas puedan ser más propensas a una clasificación que a otra (sirve como discriminante o corrección).

Mediante esta afirmación, se construye la siguiente ecuación para obtener la evidencia de que una imagen se corresponda a un tipo de terminado de dígito:
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples7.png" alt="Example 7">
</p>

Donde:

 - <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples8.png" alt="Example 8"> Es el peso de un determinado dígito
 - <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples9.png" alt="Example 9"> Sesgo de un determinado dígito
 - <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples10.png" alt="Example 10"> Índice que corresponde con un dígito
 - <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples11.png" alt="Example 11"> Índice de cada píxel de una imagen
