"""
pick.py es un módulo con una única clase: PickU2Net, que determina puntos de agarre para objetos en una imagen.  Típicamente se importa así:

  .. code-block:: python

    import PickU2Net from pick

Usa el modelo U2Net entrenado para segmentar objetos en la imagen de entrada, analiza los contornos y devuelve los puntos de agarre para cada objeto encontrado.

Los puntos de agarrre corresponden a grippers de dos dedos: antipodal grasping.

Se usa sólo con imágenes de una cámara común, y devuelve posiciones en 2D sobre las mismas.
No determina la profundidad.

El módulo se puede ejecutar como script para probarlo con una imagen de ejemplo de esta manera:

  .. code-block:: bash

    python pick.py --input images/imagen_13r.jpeg  --model u2net

Recomendación: en la imagen conviene que el fondo sobresalga por todos los bordes.

El script muestra dos ventanas con la imagen segmentada obtenida por el modelo y la imagen anotada con los puntos de agarre.

Pruebe también el parámetro --model u2netp para usar el modelo liviano, más rápido y menos preciso.

"""

from u2net_predict import U2netModel
import numpy as np
import cv2 as cv
from types import SimpleNamespace

class PickU2Net:
  """
  PickU2Net determina puntos de agarre para objetos en una imagen.

  - El constructor configura el objeto. 
  - El método __call__ analiza la imagen de entrada y obtiene los puntos de agarre.
  - El método annotate anota la imagen de entrada con los puntos de agarre.
  - Otros métodos son usados internamente.

  Muchos resulados intermedios y finales son almacenados en atributos del objeto para poder accederlos sin repetir las operaciones.

  Ejemplo de uso:

    .. code-block:: python

        picking = PickU2Net()
        results = picking(input_image)

  El método __call__ documenta los resultados devueltos.

  Attributes:
      model (U2netModel): Modelo U2Net para segmentar la imagen.
      minArea (int): Área mínima de un contorno para ser considerado.
      maxArea (int): Área máxima de un contorno para ser considerado.
      threshold (int): Umbral para segmentar la imagen.
      map (np.ndarray): Máscara binaria de la imagen segmentada.
      contours (list): Lista de contornos encontrados en la imagen.
      results (list): Lista de objetos con los puntos de agarre y otros datos de cada objeto encontrado.      

  """
  def __init__(self, model_name:str="u2net", minArea:int=100, maxArea:int=100000, threshold:int=128):
    """Inicializa el objeto PickU2Net con los argumentos opcionales.
    
    Args:
        model_name (str, optional): Nombre del modelo a usar, u2net o u2netp. Por defecto "u2net".
        minArea (int, optional): Área mínima de un contorno para ser considerado. Por defecto 100.
        maxArea (int, optional): Área máxima de un contorno para ser considerado. Por defecto 100000.
        threshold (int, optional): Umbral para segmentar la imagen. Por defecto 128.
    """
    self.model = U2netModel(model_name)
    self.minArea = minArea
    self.maxArea = maxArea
    self.threshold = threshold

  def __call__(self, input_image:np.ndarray)->list[SimpleNamespace]:
    """
    Procesa la imagen dada y devuelve los puntos de agarre.

    Usa U2Net para segmentar la imagen argumento, analiza los contornos y devuelve los puntos de agarre para cada objeto encontrado.

    Invoca los métodos analizeContour y getGrabPoints para cada contorno.
    
    Args:
        input_image (np.ndarray): Imagen de entrada

    Returns:
        list[SimpleNamespace]: Lista de objetos con los puntos de agarre y otros datos de cada objeto encontrado.
          Cada elemento de la lista corresponde a un objeto detectado, y tiene las siguientes propiedades:
            - center: tuple[int,int], coordenadas en píxeles del baricentro del contorno
            - principalComponent: np.array, versor 2D apuntando en la dirección del componente principal
            - grabbingPoint0: tuple of int, punto de agarre
            - grabbingPoint1: tuple of int, otro punto de agarre
            - contour: np.array, contorno analizado

          Los elementos None corresponden a contornos rechazados.
  
    """
    output_image = self.model(input_image)
    self.map = cv.inRange(output_image, self.threshold, 255)  # You can adjust threshold (inRange 2nd argument)
    self.contours, self.hierarchy = cv.findContours(self.map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # cv.RETR_CCOMP for 2 level (contours with their holes)
    self.results = []

    # Analize and annotate contours
    for contour in self.contours:
      # Filter contours
      area = cv.contourArea(contour)
      if(area<self.minArea or area>self.maxArea):
        # Too small or too big
        self.results.append(None)
        continue

      # gravity center c, principal component unity vector v
      center, principalComponent = self.analizeContour(contour)
      grabbingPoint0, grabbingPoint1 = self.getGrabingPoints(contour, center, principalComponent)
      result = SimpleNamespace()
      result.center = center
      result.principalComponent = principalComponent
      result.grabbingPoint0 = grabbingPoint0
      result.grabbingPoint1 = grabbingPoint1
      result.contour = contour
      self.results.append(result)

    return self.results
  

  def annotate(self, input_image:np.ndarray)->np.ndarray:
    """Anota la imagen con los puntos de agarre.

    Convierte la imagen a blanco y negro y para cada objeto detectado anota:
      - puntos de agarre
      - centro
      
    Dibuja los contornos, los puntos de agarre en la imagen de entrada.


    Args:
        input_image (np.ndarray): Imagen de entrada

    Returns:
        np.ndarray: Imagen anotada

    """
    # Gray image for color annotation
    imVis = cv.cvtColor(cv.cvtColor(input_image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

    # Analize and annotate contours
    for result in self.results:
      if(not result):
        continue

      # Annotation
      cv.drawContours(imVis, [result.contour], 0, (0, 128, 255), 2)
      arrow = (result.principalComponent*50).astype(np.int32)
      cv.arrowedLine(imVis, result.center, result.center + arrow, (255, 128, 0), 3, tipLength=0.2)

      cv.circle(imVis, result.grabbingPoint0, 5, (0,0,255), 3)
      cv.circle(imVis, result.grabbingPoint1, 5, (0,0,255), 3)

    return imVis

  def analizeContour(self, contour:np.ndarray)->tuple[tuple[int,int],np.ndarray]:
    """
    Obtiene baricentro y componente principal de un contorno.

    #. Computa los momentos del contorno.
    #. Obtiene el baricentro con los momentos de primer orden.
    #. Calcula el componente principal como primer autovector de la matriz de covarianza obtenida con los momentos centrales de segundo orden.
  
    Args:
        contour (np.ndarray): Contorno a analizar

    Returns:
        tuple[tuple[int,int],np.ndarray]: Baricentro y componente principal

    """

    moments = cv.moments(contour)
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    M = np.array([[moments['mu20'], moments['mu11']],[moments['mu11'], moments['mu02']]])
    eigenvalues, eigenvectors = np.linalg.eig(M)  # Always two eigen
    if(eigenvalues[0]>eigenvalues[1]):
      principalComponent = eigenvectors[:, 0]
    else:
      principalComponent = eigenvectors[:, 1]
    return center, principalComponent

  def getGrabingPoints(self, contour:np.ndarray, center:tuple[int,int], principalComponent:float)->tuple[tuple[int,int],tuple[int,int]]:
    """
    Obtiene dos puntos de agarre en un contorno a partir del resultado de ``analyzeContour()``
    
    Dado un contorno, un baricentro y un componente principal, calcula dos puntos de agarre.

    El primer punto se encuentra en la intersección del contorno con la recta perpendicular al componente principal que pasa por el baricentro.
    
    El segundo punto se encuentra en la intersección del contorno con la misma recta, pero en el lado opuesto del baricentro.
    
    No chequea si más de dos puntos son intersectados, ni si los puntos de contacto no son normales a los dedos del gripper.
    
    Estas dos verificaciones pendientes podrían ser implementadas en futuras versiones.

    Args:
      contour (np.ndarray):
      center (tuple[int,int]):
      principalComponent (float):
    
    """
    
    contour = np.squeeze(contour)
    distancesFromEdge = np.dot(contour - center, principalComponent)
    zeroDistanceIndices = np.argwhere(np.abs(distancesFromEdge)<1.0)
    middleIndex = int((zeroDistanceIndices[0]+zeroDistanceIndices[-1])/2)

    p0Indices = zeroDistanceIndices[zeroDistanceIndices < middleIndex]
    p0Points = contour[p0Indices]
    p0 = np.average(p0Points, axis=0).astype(np.int32)

    p1Indices = zeroDistanceIndices[zeroDistanceIndices > middleIndex]
    p1Points = contour[p1Indices]
    p1 = np.average(p1Points, axis=0).astype(np.int32)
    
    return p0,p1


# Main
if __name__ == "__main__":
  import argparse

  # Parsing arguments from command line
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="input image file path", default="images/imagen_13r.jpeg")
  parser.add_argument("-m", "--model", help="model, either u2net (default) or u2netp", default="u2net")
  args = parser.parse_args()

  model_name = args.model
  input_image_path = args.input
  #output_image_path = args.output

  # Read image and analyze
  input_image = cv.imread(input_image_path)
  picking = PickU2Net(model_name)
  results = picking(input_image)
  imVis = picking.annotate(input_image)

  # Show results
  cv.imshow('Map', picking.map)
  cv.imshow('Grasp', imVis)
  cv.waitKey(0)