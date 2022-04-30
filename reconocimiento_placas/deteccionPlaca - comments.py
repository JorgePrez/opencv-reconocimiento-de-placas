# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:12:54 2022

@author: opere
"""

#LA SALIDA DE ESTE SCRIPT SERÍA LA PLACA CON CORRECIÓN DE PERSPECTIVA


import cv2
import numpy as np
from scipy import ndimage




def detectarPlaca(img):
    
    I=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    u,_=cv2.threshold(I,0,255,cv2.THRESH_OTSU)
    
    
    mascara = np.uint8(255*(I>u))
    
    #etiquetas ibjetos para analizarlos
    
    output=cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
    
    cantObj=output[0]
    labels=output[1]
    stats=output[2]
    maskObj=[]  #guradar cada objeto separado y eteiquetado en esta lista, la 
    #la idea es determinar cual de esos objetos es la placa
    
    
     
    maskConv=[] 
    diferenciaArea=[] #comparar el convex hull con el arae de la mascara original apra ver si se parece o no
    
    
    
    for i in range(1,cantObj):
        if stats[i,4]>stats[:,4].mean(): #/10: 
            #condicional del area, no tiene sentido procesar objetos pequeños, area es mayor que el promedio
           # sacar todos los valroes de area y calcuar el promedio, analizar solo si el area del objeto actual es mayor que el promedio
            
            mascara=ndimage.binary_fill_holes(labels==i) #rellanndo huecos de cada imagen individual
            mascara=np.uint8(255*mascara)  #de arreglo booleano a entero de 8 bits
            maskObj.append(mascara)
            
            #para saber si un objeto es una placa o no, tendriamos que utilzar la forma del objeto
            
            #comprar el area del objeto con el area del convex hall,
            
            #EL AREA DEL CONVEX HULL DE UNA PLACA, ya que la placa es convexa, serpia muy parecido
            
            #calculo convexhull
            _,contours,_=cv2.findContours(mascara,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
            cnt=contours[0]
            hull=cv2.convexHull(cnt)
            
            puntosConvex=hull[:,0,:]
            m,n=mascara.shape
            ar=np.zeros((m,n))
            
            mascaraCovex=np.uint8(255*cv2.fillConvexPoly(ar,puntosConvex,1))
            maskConv.append(mascaraCovex)
            
            #comparacion area CH con objeto
            areaObj=np.sum(mascara)/255
            areaConv=np.sum(mascaraCovex)/255
            diferenciaArea.append(np.abs(areaObj-areaConv))
            
    maskPlaca=maskConv[np.argmin(diferenciaArea)]   #la posicion donde se cumpla el minimo sera la placa
        
    ### correcion de perspectiva
    
    #a partir de mascara detectar los vertices, cual es la posicion del os vertices
    # y hallar otros nuevos vertices donde quiero que se vayan estos puntos
    
    
    vertices=cv2.goodFeaturesToTrack(maskPlaca,4,0.01,10)
    x=vertices[:,0,0] #devuelve un arreglo no de 4x2 si uno de 4x1x2 (por eso el cero)  y asi tenemos 4 valroes en x , 4 en y
    y=vertices[:,0,1] 
    
    #hallar otros 4 putnos que esten alienados con los ejes
    vertices=vertices[:,0,:] # vertices queda de 4x2
    
    #ordenando pos:0 mayor y la ultima menor
    xo=np.sort(x) 
    yo=np.sort(y)
    
    
    #nuevos vertices
    xn=np.zeros((1,4))
    yn=np.zeros((1,4))
    
    
    # n: distancia entre maximo y minimo 
    n=(np.max(xo)-np.min(xo))
    
    m=(np.max(yo)-np.min(yo))
    
    xn=(x==xo[2])*n+(x==xo[3])*n
    yn=(y==yo[2])*m+(y==yo[3])*m
    
    verticesN=np.zeros((4,2))
    verticesN[:,0]=xn
    verticesN[:,1]=yn
    
    vertices=np.int64(vertices)
    verticesN=np.int64(verticesN)
        
    h,_=cv2.findHomography(vertices,verticesN)
        
    #plcaca con correction de perspectiva 
    
    placa=cv2.warpPerspective(img,h,(np.max(verticesN[:,0]),(np.max(verticesN[:,1]))))
    
    return placa


#img=cv2.imread("placa1.jpg")
img=cv2.imread("car6.jpg")
placa=detectarPlaca(img)
  

#cv2.imshow("",placa)
 

    
        
        
        

