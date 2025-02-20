'''
Servidor que escucha consultas TCP provenientes del robot, solicitando coordenadas de picking.

Incluye un sistema de visión artificial provisto por la biblioteca pick, 
que determina las coordenadas de picking en una imagen.

Incluye también una rutina de calibración extrínseca para la cámara que determina la matriz de homografía a partir de un patrón de calibración ajedrez de 6x9.

El sistemas de visión artificial puede detectar varios objetos en la imagen.
El servidor responde con las coordenadas de picking de un objeto a la vez.

El sistema se controla con teclas:

- C: calibrar la cámara con un patrón ajedrez de 9x6 (es lo primero que hay que hacer al ejecutar)
- D: detectar piezas, decide el agarre de cada una y las informa al robot de a una por vez
- espacio: borrar piezas
- esc: salir

El servidor se puede probar con un cliente para test que hace el rol del robot.

En desarrollo, no funcional aún.
'''

import socket
import threading
import cv2 as cv
import numpy as np
from pick import PickU2Net
from lib.extrinsics import ExtrinsicCalibrator

def get_my_ip_address(test="8.8.8.8"):
    """
    Releva el nº de IP de la propia máquina.

    Para eso abre un socket contra un nº de IP (por defecto un dns de Google).

    Returns:
        (string) ip de la propia máquina
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    myIP = ""
    try:
        sock.connect((test, 80))
        myIP = sock.getsockname()[0]
    finally:
        sock.close()
        return myIP

def start_server():
    """
    Función que ejecuta el servidor TCP/IP en un hilo aparte

    Protocolo:
    - El servidor recibe una consulta en formato de string, que no se analiza
    - La string de respuesta tiene el fomato requerido por asciiToFloat en el robot, con esta información:
        
    ``({x}, {y}, {angle} ,{aperture}, {1.0 if object_detected else 0.0})\\n``

    """

    global object_detected, objects

    # Crear un socket TCP/IP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print(f"Servidor escuchando en {HOST}:{PORT}...")

        # Bucle infinito para aceptar múltiples conexiones
        while True:
            # Usamos un timeout para evitar bloqueo completo en accept
            # Espera recibir datos hasta un segundo, y luego desbloquea con un error de timeout
            s.settimeout(1.0)
            try:
                # accept() bloquea hasta que recibe una consulta
                # addr contiene la ip de la máquina que realiza la consulta
                connection, addr = s.accept()
            except socket.timeout:
                # Si no hay conexión, seguimos esperando
                continue

            with connection:
                # Conexión establecida, recibir la consulta del cliente, una string que se ignora
                data = connection.recv(1024)
                consulta = data.decode()
                print(f"Consulta: {consulta}")

                if(len(objects)>0):
                    # Responder con las coordenadas de picking
                    object = objects.pop()
                    
                    # Proyectar las coordenadas con H, y luego calcular el centro y el ángulo
                    # La apertura se informa en mm, la mayoría de las aplicaciones del robot no la usan
                    p0 = projectPoint(extrinsicCalibrator.H, object.grabbingPoint0)
                    p1 = projectPoint(extrinsicCalibrator.H, object.grabbingPoint1)
                    center = (p0 + p1) / 2
                    angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0]) * 360.0 / (2.0 * np.pi)
                    aperture = np.linalg.norm(p1 - p0)

                    # Enviar las variables al cliente con el fomato requerido por asciiToFloat
                    response = f"({center[0]}, {center[1]}, {angle} ,{aperture}, 1.0)\n"

                else:
                    # No hay objetos detectados
                    response = f"(0.0, 0.0, 0.0, 0.0, 0.0)\n"

                connection.sendall(response.encode())
                print(f"Respuesta: {response}")

                # Reiniciar la variable de objeto detectado
                object_detected = False

                # Respuesta enviada. Conexión cerrada. Esperando nueva conexión...

def projectPoint(H, point):
    """
    Proyecta un punto usando la homografía.

    # Expande el punto dado a coordenadas homogéneas en el espacio proyectivo asociado
    # Lo proyecta con la transformación lineal H
    # Normaliza el resultado y lo reduce a 2 dimensiones devolviéndolo al espacio vectorial

    Args:
        H: homografía, matriz de 3x3
        point: punto 2D

    Returns:
        punto 2D proyectado
    """
    # Proyectar el punto con la homografía
    p = np.array([point[0], point[1], 1.0])
    p = np.dot(H, p)
    p = p / p[2]
    return p[0:2]

if __name__ == "__main__":
    print('''
    Servidor TCP/IP para comunicación con robot de picking.
    El servidor recibe consultas del robot y responde con las coordenadas de picking.
    Preparacón:
        - Calibrar extrínsecamente la cámara con un patrón ajedrez de 9x6 casillas (con la tecla C)
        - Detectar la posición de las piezas con la tecla D

    Teclas:
        C: Calibrar extrínsecamente (computa la homografía)
        D: Detectar la posición de la pieza
        espacio: Borrar las detecciones
        ESC: Salir del bucle
        Ctrl+C: Detener el servidor y salir del programa
    ''')

    # Coordenadas de picking que se informarán al robot
    object_detected = False
    objects = []

    # IP y puerto de escucha de este servidor TCP/IP
    HOST = get_my_ip_address() # Dirección IP de este servidor
    PORT = 65432           # Puerto donde escucha el servidor, un número inventado que no esté usado por otro servicio

    # Crear un hilo para el servidor
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True  # Hacer que el hilo se cierre cuando termine el programa principal
    server_thread.start()

    # Programa principal continúa haciendo otras cosas
    print(f'Servidor escuchando en IP {HOST} y puerto {PORT}.')

    extrinsicCalibrator = ExtrinsicCalibrator()
    pickU2Net = PickU2Net()
    cam = cv.VideoCapture(0)


    # Bucle principal donde se determinan las coordenadas x e y
    try:
        while True:
            im = cam.read()

            # Pausa pequeña para permitir que el servidor siga atendiendo consultas
            key = cv.waitKey(100)
            if key == 27:
                break

            key = ord(key)
            if key == 'c':
                # Calibrar
                if extrinsicCalibrator.findCorners(im):
                    extrinsicCalibrator.computeHwc()
                    print(f"Hwc: {extrinsicCalibrator.Hwc}")

                    Hviz = extrinsicCalibrator.getHviz(scaleFactor=25.0, translation=(5,5))
                    imFrontal = cv.warpPerspective(im, Hviz, (im.shape[1], im.shape[0]))
                    cv.imshow('Frontal', imFrontal)
                    im = extrinsicCalibrator.drawCorners()

                else:
                    print("No se detectó el patrón")

                cv.imshow('Cam', im)


            elif key == 'd':
                # Detectar objetos
                objects = pickU2Net(im)

            elif key == ' ':
                # Borrar detecciones
                objects = []

    except KeyboardInterrupt:
        print("Programa interrumpido por el usuario.")