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
import imutils

#Directorio en donde estaran las imagenes que se obtienen de la webcam
personName = input('Ingrese su nombre: ')
dataPath = './Parcial3IA/data'
personPath = dataPath + '/' + personName

#Si la carpeta con el nombre del usuario no existe, entonces se crea la carpeta
if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)

#Variable para invocar el metodo que prende la webcam
cap = cv2.VideoCapture(0)

#Clasificador de Haar para detectar el rostro y la sonrisa
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
count = 0

#Ciclo que añade los cuadros en el rostro y la sonrisa de acuerdo a los parametros
#del clasificador de Haar, se procesan las imagenes en escala de grises debido que 
#asi el clasificador las reconoce mejor y las almacena en la carpeta data
while True:
    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    # 1.3 es el factor de escala
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    
    #Los datos faciales se almacenan como tuplas de coordenadas. 
    #Aquí, x e y definen las coordenadas de la esquina superior izquierda del marco de la cara
    #w y h definen el ancho y la altura del marco.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
  
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color,(sx, sy),((sx + sw),(sy + sh)),(0, 0, 255), 2)

        count = count + 1

    cv2.imshow('frame',frame)
    k =  cv2.waitKey(1)

    #Se debe tener apretada la tecla q para cerrar la webcam mientras se ejecuta el
    #reconocedor de rostro o esperar hasta que el limite de imagenes llegue a 100, en
    #ese entonces el programa termina
    if cv2.waitKey(1) & 0xff == ord('q') or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()