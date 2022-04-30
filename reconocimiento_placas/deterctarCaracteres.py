# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:19:48 2022

@author: opere
"""



import cv2
import numpy as np
from scipy import ndimage
from deteccionPlaca import detectarPlaca
from reconocimientoCaracteres import clasificadorCaracteres, get_hog, escalar


#img=cv2.imread("blanco-a.jpg") 

#img=cv2.imread("blanco-f.jpg") 

#img=cv2.imread("azul-a.jpg") 

img=cv2.imread("azul-f.jpg")

#img=cv2.imread("car6.jpg")
placa=detectarPlaca(img)



I=cv2.cvtColor(placa,cv2.COLOR_BGR2GRAY)

u,_=cv2.threshold(I,0,255,cv2.THRESH_OTSU)

mascara=np.uint8(255*(I<u))
output = cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
cantObj=output[0]
labels=output[1]
stats=output[2]




for i in range(1,cantObj):
    if stats[i,4]<stats[:,4].mean()/10:  #objeto menor a las 10 fraccion del promedio del area
        labels=labels-i*(labels==i) 
        
    if stats[i,4]>stats[:,4].mean()*1.5:  #objeto que no coincide con las dimensiones promedio
        labels=labels-i*(labels==i)#etiqueta que no interesa , es quitada 
        
mascara=np.uint8(255*(labels>0))
kernel=np.ones((3,3),np.uint8)
#dilatar imagen
mascara=np.uint8(255*ndimage.binary_fill_holes(cv2.dilate(mascara,kernel)))

#hallar bounding box recto por caracteres

_,contours,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
caracteres=[]
orden=[] #segun posicion en X
placa2=placa.copy()
for cnt in contours:
    x,y,w,h=cv2.boundingRect(cnt)
    caracteres.append(placa[y:y+h,x:x+w,:])  #cv2.imshow("",caraceteres[0])
    orden.append(x) 
    cv2.rectangle(placa2,(x,y),(x+w,y+h),(0,0,255),1) #bgr
    
#ordenar caracteres para que queden en la misma direccion que la placa    
caracteresOrdenados = [x for _,x in sorted(zip(orden,caracteres))]

palabrasKnn=""
palabrasSVM=""
hog=get_hog()
knn,SVM=clasificadorCaracteres()

posiblesEtiq=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

posiblesEtiq=np.array(posiblesEtiq)

caracteresPlaca=[]

for i in caracteresOrdenados:
    m,n,_=i.shape
    escalado=escalar(i,m,n)
    caracteresPlaca.append(escalado) # cv2.imshow("",caracteresPlaca[0])
    caracteristicas = np.array(hog.compute(escalado))
    palabrasKnn+=posiblesEtiq[knn.predict(caracteristicas.T)][0][0] #.t trasnponer 1x144 asi lo recibe el clasificador
    palabrasSVM+=posiblesEtiq[SVM.predict(caracteristicas.T)][0][0]

print("El clasificador Knn dice: "+palabrasKnn)    
print("El clasificador SVM dice: "+palabrasSVM)   #funciona mejor para este tipo de problemas
  



m,n,_=img.shape
cv2.putText(img,"La placa es: "+palabrasSVM,(10,300),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,255),3)

cv2.imshow("carro",img)
cv2.imshow("placa",placa2)








