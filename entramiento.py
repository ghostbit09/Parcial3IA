'''
Integrantes del grupo:
    - Braian Camilo Piedrahita Rodriguez
    - Sebastian Quintero Osorio
    - Melissa Ortiz Perez
    - Fernando Jose Murcia Hincapie

    Para que funcione el proyecto hay que instalar estas librerias:

    pip install opencv-contrib-python --no-cache-dir
    pip install imutils
'''
import cv2
import os
import numpy as np

#Ruta en donde se encuentran las carpetas con las imagenes de los rostros
dataPath = './Parcial3IA/data'
#Lista con los nombres de las carpetas de las imagenes de los rostros de cada persona
peopleList = os.listdir(dataPath)
#Se imprime la lista para verificar si esta detectando las carpetas
print('Lista de personas: ', peopleList)
#Lista que tendra los identificadores de cada persona
labels = []
#Lista que tendra los datos de los rostros de cada persona
facesData = []
#Variable para almacenar los identificadores en el ciclo
label = 0

#Ciclo para recorrer la carpeta de cada persona (Usuario)
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir #Ruta completa de la carpeta del usuario
    print('Leyendo las imágenes')

    #Ciclo para recorrer las imagenes en cada carpeta
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        #Se añade el identificador de la persona por imagen a la lista de labels
        labels.append(label)
        #Se añade cada una de las imagenes en escala de grises con OpenCV
        #Con el parametro 0 se indica la transformacion a escala de grises
        facesData.append(cv2.imread(personPath+'/'+fileName,0))

        #Para visualizar la lectura de las imagenes en escala de grises
        #image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    label = label + 1

#Para visualizar si se agregaron los datos a las listas
#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

#Aplicamos el entrenamiento con la libreria de EigenFaces
face_recognizer = cv2.face.EigenFaceRecognizer_create()

print('Entrenando el reconocedor...')

#Se pasa al entrenamiento del reconocedor utilizando los datos obtenidos anteriormente
#tranformando el arreglo con los identificadores en un numpy array, esto puede tomar
#tiempo debido a la cantidad de datos (imagenes) que tiene cada carpeta
face_recognizer.train(facesData, np.array(labels))

#Se obtiene el modelo obtenido del entrenamiento
face_recognizer.write('./Parcial3IA/modeloReconocedor.xml')

print('Modelo almacenado correctamente')