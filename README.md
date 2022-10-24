# Aprendizajes automáticos. TP-3


1. Resuelva los siguientes ıtems:
    - a) Construya un conjunto "TP3-1" de ejemplos linealmente separables en R2 (por ejemplo, en forma aleatoria, genere puntos en 
    [0, 5]x[0, 5] tal que pertenezcan a dos clases linealmente separables 1 y −1). Utilice un perceptron simple escalón para separar al conjunto linealmente. ¿El hiperplano de separación es óptimo? Justifique la respuesta.
    - b) Obtener el hiperplano óptimo a partir del hiperplano que obtuvo con el perceptronsimple.
    - c) Construya el conjunto "TP3-2" de forma análoga a como construyó el conjunto "TP3-1" pero esta vez incluya algunos ejemplos que queden mal clasificados cerca del hiperplano de separación. Utilizar un perceptron simple para separar las dos clases. Comentar los resultados.
    - d) Utilice SVM para clasificar tanto el conjunto "TP3-1" como el conjunto "TP3-2". Compare los resultados con los obtenidos en el punto b) y c).

2. Segmentación de imágenes en color: Considere la imagen cow.jpg y las imágenes muestra: vaca.jpg, cielo.jpg y pasto.jpg correspondientes a las clases “vaca”, “cielo” y “pasto”, respectivamente.
    - Construir un conjunto de datos para entrenamiento, indicando para cada muestra a qué clase pertenece.
    - Dividir aleatoriamente el conjunto de datos en dos conjuntos, uno de entrenamiento y uno de prueba. Utilizar el método SVM para clasificar los pixels del conjunto de prueba, entrenando con el conjunto de entrenamiento.
    - Con el mismo método ya entrenado clasificar todos los pixels de la imagen. 
    - Clasificar una imagen similar.
