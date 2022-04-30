# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 00:26:18 2022

Reconocimiento de caracteres : entrenar y definir los clasificadores 

@author: opere
"""


import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.manifold import TSNE



#####Funcion donde se definen los parametros de los descriptores de hog


def get_hog():
    winSize = (20,20) #tenemos imagenes de 10x20 y de 20x20, como es tamaño fijo se debera hacer un resize
    blockSize=(8,8) # definiendo CELDAS
    blockStride = (4,4) # el paso que va a dar , cada cuenato se mueve la ventana
    cellSize=(8,8) 
    nbins = 9  #cantidad de barras a utilizar 
    derivAperture = 1   #parametros de convergencia
    winSigma = 2.   #filtro al principio para la imagen para promediarla , y elimianr el ruido que puede haber
    histrogramType = 0  #default
    L2HysThreshold = 0.2 # una constante por la que se multiplcia la magnitud al momento de normalizarse, SIRVE para reducir todavia más los valores...  
    gammaCorrection = 1 #preprocesamiento para mejorar caracteristicas
    nlavels = 64  #default
    signedGradient = True   #si queremos que el gradiente sea con signo? 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histrogramType,L2HysThreshold,gammaCorrection,nlavels,signedGradient)
    return hog



######escalar imagen a dimensiones de 20x20, necesario para la funcion siguiente
    
    
def escalar(img,m,n):  #agregar valores blancos a los lados, se añadiran arriba o a los lados dependiendo de la condición, el objetivo es que la imagen quede simetrica cuadrada
    
    if m>n:
        imgN=np.uint8(255*np.ones((m,round((m-n)/2),3))) #creando imagen en blanco
        escalada=np.concatenate((np.concatenate((imgN,img),axis=1),imgN), axis=1)  #concater 2 veces, izquierdo, derecho

    else:
        imgN=np.uint8(255*np.ones((round((n-m)/2),n,3))) # 
        escalada=np.concatenate((np.concatenate((imgN,img),axis=0),imgN), axis=0) # concatenar arriba y abajo
    
    img = cv2.resize(escalada, (20,20))
    
    return img
    


######generacion de (base de datos de) caracteristicias


def obtenerDatos():
    
    #posiblesEtiquetas son las clases, no se toman en cuenta la O , ni la i, ni la ñ (no estan en la base de datos)
    posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    datos = [] #caracteristicas
    etiquetas = []
    
    
    for i in range(1,26):
            for j in posiblesEtiq:
                img=cv2.imread(j+'-'+str(i)+".jpg")
                if img is not None:
                    m,n,_=img.shape
                    if m !=20 or n !=20:
                        img =escalar(img,m,n)
                    etiquetas.append(np.where(np.array(posiblesEtiq)==j)[0][0]) #devuelve la posicion A=10 B=11 .... mas util tenerlo como numero, favorece a problemas multilclase
                    #se coloca [0] [0] encapsulado arreglo por ejemplo: 600 etiqutes , pero con este metedo append/where aparece 600[1][1]  
                    hog = get_hog()
                    datos.append(np.array(hog.compute(img))) #calcular las caracteristicas de la img(ya procesadas)
                    #36 caracteristicas por cada 4 celdas , los pasos eran de 4( 2 pasos en cada eje)
                    #144=9*4*2*2  (9 barrras)(4 vecinos)() 
    
    datos=np.array(datos)[:,:,0] #al convertilo a array de numpy devuelve (697,144,1), eliminado esta ultima dimension
    etiquetas=np.array(etiquetas) 
    return datos, etiquetas  #base de datos de caracteristicas, a que equivale cada una

    

#######Entrenando el sistema,clasificadores knn y SVM
    

def clasificadorCaracteres():
    datos, etiquetas=obtenerDatos()
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(datos,etiquetas)
    SVM=svm.SVC(kernel='linear', probability=True, random_state=0,gamma='auto')
    SVM.fit(datos,etiquetas)
    return knn, SVM


"""
     
datos, etiquetas = obtenerDatos()

#--------------------------Evaluacion KNN 
##implementando clasificador de k-vecinos cercanos, clasificador no lineal 

X_train,X_test,y_train,y_test=train_test_split(datos,etiquetas,test_size=0.2, random_state=np.random)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train) #entrenando el sistema

errorEntrenamientoKnn=(1-knn.score(X_train,y_train))*100 #knn_score devuelve el inverso del error
print("Error de entrenamiento del Knn es: "+str(round(errorEntrenamientoKnn,2))+"%")


errorPruebaKnn=(1-knn.score(X_test,y_test))*100
print("Error de prueba del Knn es: "+str(round(errorPruebaKnn,2))+"%")


# Cross-Validation


prediccionKnn=knn.predict(X_test)

errorKnn=100*(1-cross_val_score(knn,datos,etiquetas,cv=10))
print("Knn cross val: "+str(round(errorKnn.mean(),2))+"+-"+str(round(errorKnn.std(),2)))

#------------------------Matriz de confusion
#plt.imshow(confusion_matrix(y_test,prediccionKnn), interpolation = "nearest")


#plt.title("Matriz de cofusion Knn")
#plt.xlabel("Prediccion")
#plt.ylabel("verdadera etiqueta")


#-------------------------SVM  , deteccion de formas suele funcionar mejor este....

SVM=svm.SVC(kernel='linear', probability=True, random_state=0,gamma='auto')


SVM.fit(X_train,y_train)
errorSVM=100*(1-cross_val_score(SVM,datos,etiquetas,cv=10))
print("SVM cross val: "+str(round(errorSVM.mean(),2))+"+-"+str(round(errorSVM.std(),2)))



#-----------------------T-SNE ,llevar a dimension 3 o 2 para visualizacion más sencilla, agrupaciones y clases solapadas 

X=TSNE(n_components=2).fit_transform(datos)
posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

#normalizar 
x_min, x_max=np.min(X,0),np.max(X,0)
X=(X-x_min)/(x_max-x_min)

for i in range(0,len(X)):
    plt.text(X[i,0],X[i,1],str(posiblesEtiq[etiquetas[i]]), color = plt.cm.Set1(3*float(etiquetas[i])/99))

"""










