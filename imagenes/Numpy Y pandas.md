  A. Numpy arreglos

    Como primer paso veremos que son los arreglos y como se pueden aplicar, de esta forma entenderemos para que son y explicaremos de que forma estos se pueden utilizar en difernetes contextos.

    Los arreglos basicamente son una estructura por la que obtendremos formas diversas en las que podremos almacenar datos del mismo tipo, vale la pena mencionar que estas colecciones de datos son sumamente eficientes tanto en su creacion como en su almacenamiento.

    existen dos tipos, vectores y matrices.
    Cada uno de ellos tiene ciertas cualidades y  caracteristicas; por ejemplo los vectores son tecnicamente matrices, sinembargo tienen 1D, asi que se podria decir que son pseudo matrices que pueden ser creadas de manera vertical u horizontal según se desee, las matrices como tal tienen 2D, ahi radica su principal diferencia.
    
    Para empezar vamos a hablar de lo mas básico de la libreria numpy, lo cual es el modo en el que se va a importar la propia libreria, es sumamente sencillo, haremos uso de una simple linea de codigo
    
    * import numpy as np
    Esta simple lina de codigo importa una gran herramienta, nos ahorrará algunos si no muchos dolores de cabeza. en caso que quieras conocer la versión de numpy, debemos usar otra simple linea de codigo.
    * np.__version__

    EJEMPLO 1°
![Importe](./imagenes/Importe%20de%20Numpy.png)
    en este caso podemos ver que la version que fue importada es la 2.0.2

  los arreglos pueden ser creados a partir de la conversion de una lista o una lista que contenga listas, por ejemplo
  * my_list = [1, 2, 3]
  * my_list
![Listas](/imagenes/Ejemplo%20de%20listas.png)

  mediante el metodo array es posible convertir una lista a un arreglo, con este metodo, podemos tambien asignar el arreglo a un tipo de dato, por otro lado el comando "a.dtipe" nos permite conocer el tipo de datos que tiene el arreglo, ademas podemos mostrar que efectivamente el dato del arreglo "a" fue guardado

  * a = np.array(my_list)
  * type(a)
  * a.dtype 
  * print(a)

![Tipo](/imagenes/Tipo%20de%20dato%20e%20impresion.png)

* reshape
  En NumPy, este metodo se utiliza para modificar la estructura dimensional de un arreglo sin cambiar el contenido de sus datos. Matemáticamente, un arreglo puede considerarse como un tensor con cierta cantidad de dimensiones y elementos. La operación de reshape consiste en reconfigurar la distribución de esos elementos en un nuevo espacio de dimensiones, manteniendo constante el número total de elementos.

![Reshape](/imagenes/Ejemplo%20n2.png)

  * concatenar
    En NumPy, la concatenación se entiende como el proceso de unir dos o más arreglos en uno solo, conservando su estructura pero ampliando su tamaño en alguna de sus dimensiones. Existen principalmente dos formas de realizar esta unión: a lo largo de las filas o a lo largo de las columnas.

    Cuando se concatena en el eje de las filas, lo que ocurre es que los arreglos se apilan uno debajo del otro, generando así un mayor número de filas pero manteniendo el mismo número de columnas. En cambio, cuando la concatenación se hace sobre el eje de las columnas, los arreglos se disponen uno al lado del otro, aumentando el número de columnas pero conservando el número de filas.

    Un aspecto importante es que, para que la concatenación sea posible, los arreglos deben coincidir en todas las dimensiones excepto en aquella en la que se está uniendo. De esta forma, NumPy asegura la coherencia en la estructura del nuevo arreglo creado.

![Concatenar](/imagenes/EJemlpo%20de%20concatenar.png)

  * operaciones basicas
    En NumPy, las operaciones básicas se refieren a aquellas que pueden aplicarse directamente sobre los arreglos, de forma similar a como se trabaja con números o con expresiones matemáticas. La gran diferencia es que estas operaciones se realizan de manera elemento por elemento, lo que significa que cada posición del arreglo es afectada de forma independiente, pero en paralelo.

    Por ejemplo, cuando se suman dos arreglos del mismo tamaño, lo que ocurre es que cada elemento del primero se combina con el elemento en la misma posición del segundo. Lo mismo sucede con la resta, la multiplicación y la división. Esta capacidad convierte a NumPy en una herramienta muy eficiente, ya que evita el uso de ciclos tradicionales y permite cálculos vectorizados, propios de un tratamiento más matemático y cercano al álgebra lineal.

    Además de estas operaciones aritméticas básicas, NumPy también incorpora funciones matemáticas elementales como la potencia, la raíz cuadrada, el logaritmo o las funciones trigonométricas, las cuales pueden aplicarse directamente sobre los arreglos y devuelven resultados con la misma forma dimensional.
    
![OperacionesBasicas](/imagenes/Codigo.png)
![OperacionesBasicas](/imagenes/ejecucion.png)



_______________________________________________________________________________________________________

  B. NumPy – Operaciones estadísticas y funciones avanzadas
  
  Las operaciones estadísticas en NumPy incluyen funciones que permiten obtener como resultado indicadores básicos de tendencia central y dispersión con pocas líneas de código. Por otro lado, las funciones avanzadas tales como arange() y linspace() facilitan la generación de secuencias numéricas que son uniformes.

  Además, NumPy cuenta con apoyo para álgebra lineal, con funciones del submódulo linalg que permiten operaciones como multiplicación de matrices, cálculo de determinantes, inversas, entre otras. Estas herramientas pueden combinar funciones para la estadística básica, generación de datos y cálculo matriciado, facilitando el trabajo con datos numéricos en Python.

      Operaciones Estadísticas
          Las funciones estadísticas permiten obtener medidas de tendencia central y dispersión, muy útiles en el análisis de datos.

        mean()
          La función `mean()` permite calcular el promedio de los elementos de un arreglo.
  ![alt text](/imagenes/MeanPromedio.jpg)

        std() 
          La función std() devuelve la desviación estándar, que mide qué tanto se dispersan los valores respecto a la media.
  ![alt text](/imagenes/StdDesviacion.jpg)

        sum()
          La función sum() suma todos los elementos del arreglo.
  ![alt text](/imagenes/SumElementos.jpg)


      Funciones avanzadas de generación de datos 
          NumPy también facilita la creación de secuencias numéricas

        arange() 
          Crea un arreglo de valores dentro de un rango definido, con un paso específico.
  ![alt text](/imagenes/ArangeValores.jpg)
        
        linspace() 
          Genera valores igualmente espaciados entre un inicio y un fin.
  ![alt text](/imagenes/LinspaceValores.jpg)
        
        random
          Permite crear arreglos con números aleatorios. Por ejemplo, valores entre 0 y 1.
  ![alt text](/imagenes/RandomNumeros.jpg)

        Algebra Lineal en NumPy
          NumPy incluye el submódulo linalg para realizar operaciones de álgebra lineal.Esto resulta esencial en ingeniería y machine learning.
          
  ![alt text](/imagenes/LinalgAlgebraLineal.jpg)



_______________________________________________________________________________________________________
C. Pandas, Dataframes y Series


En un contexto de manipulacion de datos nos damos cuenta de que es una de las tareas mas importantes de la ciencia de datos, analisis estadistico y la inteligencia artificial. Mientras los conjuntos de datos se vuelven mas grandes y complejos, surge una necesidad de utilizar diferentes herramientas que nos permitan gestionar de una manera mas eficiente, rapida y flexible. 

En este contexto, Pandas se ha convertido en una de las librerias que mas se utilizan en Python. Su nombre proviene de Panel Data (datos en panel). Pandas ofrece estructuras de datos y funciones diseñadas especificamente para trabajar con informacion que esta organizada en diferentes filas y columnas que son similares a hojas de calculo en excel o tablas de bases de datos. 

Ventajas de Pandas: 
-Cargar y exportar diferentes datos en multiples formatos como : CSV, Excel, JSON, SQL, entre otros. 
-Nos ofrece estructuras de datos como Series y Dataframes que nos facilitan la organizacion y el analisis de los datos.
-Facilita la limpieza, transformacion y manipulacion de datos. 
-Se integra con otras librerias como NumPy, Matplotlib y Scikit-learn, lo que hace que esta herramienta sea indispensable en proyectos de analisis de datos. 

Series en Pandas: 
Una Serie en pandas es una estructura de datos unidimensionales como numeros, cadenas de texto, valores booleanos, fechas y otro tipo de informaciones. 
Lo mas importante de las Series en pandas es que cada uno de sus elementos esta asociado a un indice, lo cual funciona como una etiqueta que identifica y da un acceso rapido a cada valor. 

Caracteristicas principales de las Series: 
-Son unidimensionales. 
-Cada uno de sus elementos esta ligado a un indice. 
-Pueden almacenar diferentes tipos de datos. 
-Son importantes para trabajar con listas de datos que requieren una identificacion clara. 

Creacion y manipulacion de Series: 

Ejemplo 1: Crear una Serie desde una lista. 

![alt text](/imagenes/Ejemplo%201%20Series%20.jpg)
 
Ejemplo 2: Serie con índices personalizados. 

![alt text](/imagenes/Ejemplo%202%20Series%20.jpg)

Ejemplo 3: Serie a partir de un diccionario. 

![alt text](/imagenes/Ejemplo%203%20Series%20.jpg)

Ejemplo 4: Serie con valores booleanos. 

![alt text](/imagenes/Ejemplo%204%20Series.jpg)

Ejemplo 5: Serie con datos numéricos generados automáticamente. 

![alt text](/imagenes/Ejemplo%205%20Series%20.jpg)

Dataframes en Pandas: 
El Dataframe es la estructura mas potente y utilizada en Pandas. Es una tabla bidimensional con filas y columnas, donde cada una de las columnas puede contener un tipo de dato distinto.
Cada fila contiene un indice que la identifica, y cada columna tiene un nombre que permite acceder a los datos de forma sencilla. Esto hace que el Dataframe sea una herrameinta muy importante para el analisis de datos tabulares. 

Caracteristicas principales de Dataframes: 
-Son bidimensionales. 
-Cada columna es la realidad de una Serie de Pandas. 
-Se admiten multiples tipos de datos en diferentes columnas. 
-Permiten operaciones de seleccion, filtrado, ordenamiento, agrupacion y combinacion de datos. 
-Pueden cargarse desde diferentes fuentes externas como CSV, Excel o bases de datos. 
Gracias a los Dataframes, Pandas permite estructurar grandes volumenes de informacion y aplicar sobre ellos operaciones estadisticas, matematicas o de transformacion con muy pocas lineas de codigo. 

Creación y manipulación de DataFrames: 

Ejemplo 1: Crear un DataFrame desde un diccionario. 

![alt text](/imagenes/Ejemplo%201%20Dataframes.jpg)

Ejemplo 2: Selección de columnas. 

![alt text](/imagenes/Ejemplo%202%20Dataframes.jpg)

Ejemplo 3: Seleccionar varias columnas. 

![alt text](/imagenes/Ejemplo%203%20Dataframes%20.jpg)

_______________________________________________________________________________________________________

D. PANDAS Y SUS OPERAACIONES AVANZADAS
¿QUE ES PANDAS?

 Pandas es una librería de Python muy poderosa para el análisis y manipulación de datos.
 Permite trabajar con DataFrames (tablas de datos con filas y columnas) de forma sencilla, similar a Excel o SQL, pero con la potencia de Python.

 Con Pandas puedes:

 -Limpiar datos (quitar nulos, duplicados, etc.).

 -Transformar y organizar información.

 -Hacer análisis estadístico.

 -Exportar e importar a distintos formatos (CSV, Excel, SQL, JSON).

  *Filtros (Filtering)

Qué es: Seleccionar filas o columnas que cumplen una condición.

Uso: Permite trabajar solo con los datos relevantes.

![alt text](/imagenes/image-1.png)
![alt text](/imagenes/imagefiltro2.png)

  *GroupBy

Qué es: Agrupar datos según una o varias columnas y aplicar funciones de agregación.

Uso: Sirve para análisis estadístico, como totales, promedios, conteos.

![alt text](/imagenes/image-2.png)
![alt text](/imagenes/imagegroupby.png)

   *Merge / Join

Qué es: Combinar dos DataFrames según columnas en común (similar a JOIN en SQL).

Uso: Integrar información de distintas tablas en una sola.

![alt text](/imagenes/image-3.pngx)
![alt text](/imagenes/imagemerge.png)


     *Manejo de valores nulos

Qué es: Controlar valores faltantes (NaN).

Uso: Limpiar los datos para análisis más confiables.

![alt text](/imagenes/datoeliminar.png)
![alt text](/imagenes/imagevalires.png)

  *Exportación / Importación

Qué es: Guardar y leer datos en distintos formatos (CSV, Excel, SQL, JSON).

Uso: Compartir, guardar resultados o cargar información externa.

![alt text](/imagenes/datoimportarrr.png)


_______________________________________________________________________________________________


