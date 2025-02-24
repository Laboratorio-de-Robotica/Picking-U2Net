"""
Módulo con funciones para transformaciones geométricas en coordenadas homogéneas.

Útiles proyecciones y cambios de sistema de referencia.
"""
import numpy as np

def similarityTransform2D(scaleFactor:float=1.0, angle:float=0.0, translation=(0.0, 0.0), H:np.ndarray=None) -> np.ndarray:
    """
    Crea una matriz de transformación 2D de similitud y opcionalmente la aplica a la homografía H.

    Crea una transformación de similitud que combina traslación, escala y rotación aplicados en ese orden.

    Si no se proporciona el argumento H, se devuelve esa matriz de similitud.
    Si se proporciona una homografía H, se le aplica la similitud y se devuelve el resultado.

    La traslación se puede considerar como un cambio de origen de coordenadas: el origen se desplaza en el sentido contrario al vector de traslación.

    Los valores por defecto de los argumentos corresponden a no aplicar esa característica.

    Args:
        scale: factor de escala
        angle: ángulo de rotación en radianes
        translation: traslación, vector 2D
        H: homografía 3x3 a la que se le aplicará la transformación

    Returns:
        matriz de transformación 3x3
    """

    T = np.array([[scaleFactor, 0, translation[0]],
                   [0, scaleFactor, translation[1]],
                   [0,     0,             1]])

    if(angle != 0.0):
        # Matriz de rotación
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        T = T @ R

    if(H is not None):
        T = T @ H
    
    return T


def projectPoint(H:np.ndarray, point) -> np.ndarray:
    """
    Proyecta un punto usando la homografía.

    #. Expande el punto dado a coordenadas homogéneas en el espacio proyectivo asociado
    #. Lo proyecta con la transformación lineal H
    #. Normaliza el resultado y lo reduce a 2 dimensiones devolviéndolo al espacio vectorial

    Args:
        H: homografía, matriz de 3x3
        point: punto 2D, tupla, lista o ndarray de 2 elementos

    Returns:
        punto 2D proyectado
    """

    # Proyectar el punto con la homografía
    p = np.array([point[0], point[1], 1.0])
    p = np.dot(H, p)
    p = p / p[2]
    return p[0:2]