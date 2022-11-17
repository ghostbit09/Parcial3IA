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

#Ruta en donde se encuentran las carpetas con las imagenes de los rostros
dataPath = './Parcial3IA/data'
#Lista con los nombres de las carpetas de las imagenes de los rostros de cada persona
peopleList = os.listdir(dataPath)

#Reconocedor de rostros de EigenFaces
face_recognizer = cv2.face.EigenFaceRecognizer_create()
#Se lee el modelo creado en el entrenamiento
face_recognizer.read('./Parcial3IA/modeloReconocedor.xml')

#Variable para invocar el metodo que prende la webcam
cap = cv2.VideoCapture(0)

#Clasificador de Haar para detectar el rostro y la sonrisa
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

#El proceso en este punto es el mismo que el que se hace a la hora de calcular el rostro
#solo que ahora se usa la prediccion de EigenFaces para detectar el rostro con el que se entreno
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro) #Prediccion de EigenFaces

        cv2.putText(frame, '{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        #Los valores mas bajos o cercanos a cero quieren decir que el rostro tiene
        #similitud con los entrenados, en mi caso (Braian) el reconocedor detectaba mi
        #rostro y arrojaba valores entre 4000 y 5000, es por esto que puse esta validacion
        #para que agregara mi nombre o el de las personas del entrenamiento si el valor esta
        #por debajo de los 3200 caso contrario quiere decir que el rostro es desconocido, en
        #esta parte toca ir probando valores
        if result[1]<6000:
            cv2.putText(frame, '{}'.format(peopleList[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame, 'Desconocido',(x,y-20),2,0.8,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
  
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color,(sx, sy),((sx + sw),(sy + sh)),(0, 0, 255), 2)

    cv2.imshow('frame',frame)
    k =  cv2.waitKey(1)

    #Para cerrar la ventana de la webcam se debe mantener apretada la tecla q
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()