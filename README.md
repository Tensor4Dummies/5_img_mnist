PARTE 5 - Reconocimiento de imágenes (MNIST)
===================

 - [Parte 1.1 - Introducción](#parte-11---introducción)
 - [Parte 1.2 - MNIST](#parte-12---mnist)
 - [Parte 1.3 - Tensores de MNIST](#parte-13---tensores-de-mnist)
 - [Parte 1.4 - Funcionamiento del modelo Softmax](#parte-14---funcionamiento-del-modelo-softmax)
 - [Parte 1.5 - Implementación en TensorFlow](#parte-15---implementación-en-tensorflow)
 - - [Parte 1.5.1 - Variables e implementación](#151---variables-e-implementación)
 - - [Parte 1.5.2 - Entrenamiento de la regresión](#152---entrenamiento-de-la-regresión)

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
 -  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples9.png" alt="Example 9"> Sesgo de un determinado dígito
  -  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples10.png" alt="Example 10"> Índice que corresponde con un dígito
   -  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples11.png" alt="Example 11"> Índice de cada píxel de una imagen

Ahora que tenemos la evidencia, convertimos dicho resultado en una probabilidad *y* de que una imagen se corresponda a un tipo de terminado (dígito) utilizando el modelo softmax de nuevo. Tras obtener las probabilidades, normalizamos el resultado para pasar de las probabilidades de *y* a la etiqueta de tipos, es decir, que una imagen se corresponde a un único tipo.

-------------
## PARTE 1.5 - Implementación en TensorFlow
Ahora que tenemos la librería MNIST definida y el modelo Softmax formalizado podemos proceder a la realización del ejemplo de forma real. Para ello utilizaremos Python y librerías avanzadas de cálculo como NumPy que nos ofrecerá la potencia de cálculo necesaria para la multiplicación de las matrices del modelo.

### 1.5.1 - Variables e implementación
Empezando por lo básico, necesitamos definir varias variables que utilizaremos en el cálculo:

```python
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
```

Las variables definidas se describen de la siguiente forma:

 1. X = No es un valor como tal, es un **placeholder** (referencia) que utilizará TensorFlow en el cálculo del proceso. Necesitamos introducir cualquier número de imágenes de MNIST, por ello necesitaremos un vector de dimensión 784. Como no sabemos el tamaño de imágenes que vamos a utilizar en la entrada lo representamos como "None", o lo que es lo mismo, una dimensión de cualquier longitud.
 2. W = **Variable** utilizada para almacenar la evidencia de que una imagen [784 píxeles] se corresponda a uno de los 10 tipos [10] de dígito, por ello la dimensión de dicha variable es [784, 10].
 3. b = **Variable** utilizada para almacenar los sesgos de los diferentes tipos (dígitos).
 4. y = Modelo softmax que multiplica los diferentes pesos por los valores de los píxeles de cada imagen y les suma el sesgo de cada tipo(como hemos definido más arriba).

Como vemos, la definición del modelo (*y*) únicamente nos lleva una línea pues TensorFlow está diseñado para hacer regresiones de forma muy sencilla.

### 1.5.2 - Entrenamiento de la regresión
Definimos dentro de la fase de entrenamiento un concepto básico que es el coste o pérdida para conseguir categorizar una red neuronal como buena o mala. Se denomina coste o pérdida pues representa lo lejos que está nuestro modelo de la red neuronal del resultado esperado. Por ello, tratamos de minimizar el error lo máximo posible.

Una forma de determinar la pérdida del modelo es la entropía cruzada (cross-entropy) que nos permite saber en qué grado se está cometiendo el error. Se define con la siguiente fórmula:
<br/>
<br/>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tensor4Dummies/5_img_mnist/master/doc/mnistExamples12.png" alt="Example 12">
</p>

Donde *y* representa la probabilidad predicha y *y`* representa la probabilidad real obtenida por el modelo (el vector one-hot de probabilidades para cada dígito). La entropía nos permite fijar un valor al nivel de desajuste de la teoría a la realidad en los resultados.

La implementación de la entropía se realiza definiendo primero un placeholder para almacenar los valores correctos:
```python
y_ = tf.placeholder(tf.float32, [None, 10])
```
Tras dicha definición se implementa la entropía cruzada como:
```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```
