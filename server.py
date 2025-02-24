'''
Servidor que escucha consultas TCP provenientes del robot, solicitando coordenadas de picking.

Incluye un sistema de visión artificial provisto por la biblioteca pick, 
que determina las coordenadas de picking en una imagen.

Incluye también una rutina de calibración extrínseca para la cámara que determina la matriz de homografía a partir de un patrón de calibración ajedrez de 6x9.

El sistemas de visión artificial puede detectar varios objetos en la imagen.
El servidor responde con las coordenadas de picking de un objeto a la vez.

El sistema se controla con teclas:

- P: activa o desactiva la detección del patrón ajedrez
- C: calibrar la cámara con un patrón ajedrez de 9x6 (es lo primero que hay que hacer al ejecutar)
- D: detectar piezas, decide el agarre de cada una y las informa al robot de a una por vez
- espacio: borrar piezas
- esc: salir

El servidor se puede probar con un cliente para test que hace el rol del robot:

- test_client.py
- telnet

Para probarlo con telnet, al conectarlo basta enviar cualquier mensaje y 
el servidor responde con las coordenadas de picking, o todos ceros si no tiene una pieza lista.

'''

import socket
import threading
import cv2 as cv
import numpy as np
from pick import PickU2Net
from lib.extrinsics import ExtrinsicCalibrator
import lib.geometricTransforms as gt

def get_my_ip_address(test:str="8.8.8.8")->str:
    """
    Releva el nº de IP de la propia máquina.

    Para eso abre un socket contra un nº de IP (por defecto un dns de Google).

    Returns:
        ip de la propia máquina
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
        
    ``({1.0 if object_detected else 0.0}, {x}, {y}, {angle} ,{aperture})\\n``

    """

    global object_detected, objects

    # Crear un socket TCP/IP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', PORT))  # ip '' acepta todas las ip de la máquina, incluyendo localhost
        s.listen()

        print(f"Servidor escuchando en {HOST}:{PORT}...")
        print(f'getsockname {s.getsockname()}')

        # Bucle infinito para aceptar múltiples conexiones
        while True:
            # accept() bloquea hasta que recibe una consulta
            # addr contiene la ip de la máquina que realiza la consulta
            connection, addr = s.accept()

            while True:
                # Conexión establecida, recibir la consulta del cliente, una string que se ignora
                data = connection.recv(1024) # Bloqueante

                if not data:
                    # el cliente cerró la conexión
                    break

                print(data)
                print(f"Consulta: {data}) #{data.decode('utf-8', errors='ignore')}")

                if len(objects)>0:
                    # Responder con las coordenadas de picking
                    object = objects.pop()
                    
                    # Proyectar las coordenadas con H, y luego calcular el centro y el ángulo
                    # La apertura se informa en mm, la mayoría de las aplicaciones del robot no la usan
                    p0 = gt.projectPoint(extrinsicCalibrator.Hwc, object.grabbingPoint0)
                    p1 = gt.projectPoint(extrinsicCalibrator.Hwc, object.grabbingPoint1)
                    center = (p0 + p1) / 2
                    angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0]) * 360.0 / (2.0 * np.pi)
                    aperture = np.linalg.norm(p1 - p0)

                    # Enviar las variables al cliente con el fomato requerido por asciiToFloat
                    response = f"(1.0, {center[0]}, {center[1]}, {angle}, {aperture})\n"

                else:
                    # No hay objetos detectados
                    response = f"(0.0, 0.0, 0.0, 0.0, 0.0)\n"

                connection.sendall(response.encode())
                print(f"Respuesta: {response}")

            # Conexión cerrada. El siguiente bucle espera nueva conexión.
            connection.close()

if __name__ == "__main__":
    print('''
    Servidor TCP/IP para comunicación con robot de picking.
    El servidor recibe consultas del robot y responde con las coordenadas de picking.

    Teclas:
        C: Calibrar extrínsecamente (computa la homografía) con un patrón ajedrez de 9x6 casillas
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

    pattern_detection_state = True
    pattern_detected = False

    # Bucle principal donde se determinan las coordenadas x e y
    while True:
        ret, im = cam.read()

        if pattern_detection_state:
            # busca el patrón ajedrez
            pattern_detected = extrinsicCalibrator.findCorners(im)
            if pattern_detected:
                im = extrinsicCalibrator.drawCorners(im)

        cv.imshow('Camara', im)

        # Pausa pequeña para permitir que el servidor siga atendiendo consultas
        key = cv.waitKey(100)
        if key < 0:
            continue
        elif key == 27:
            break

        key = chr(key)
        if key == 'p':
            pattern_detection_state = not pattern_detection_state
            print(f'Detección del patrón de calibración: {pattern_detection_state}')

        elif key == 'c':
            # Calibrar
            if pattern_detection_state and pattern_detected:
                Hwc = extrinsicCalibrator.computeHwc()
                print(f"Hwc: \n{Hwc}")
                pattern_detection_state = False

                # Homografía para visualización: los argumentos de getHviz son arbitrarios y se deberían ajustar al caso particular
                Hviz = gt.similarityTransform2D(scaleFactor=25.0, translation=(320,240), H=Hwc)
                imFrontal = cv.warpPerspective(im, Hviz, (im.shape[1], im.shape[0]))

                cv.imshow('Frontal', imFrontal)

            else:
                if not pattern_detection_state:
                    print('Primero pulse P para activar la detección del patrón de calibración.')
                else:
                    print('No se detectó el patrón.')


        elif key == 'd':
            # Detectar objetos
            objects = pickU2Net(im)
            cv.imshow('u2net', pickU2Net.map)

            im_picking = pickU2Net.annotate(im)
            cv.imshow('Agarre', im_picking)

            print(f'Detecciones: {len(objects)}')

        elif key == ' ':
            # Borrar detecciones
            objects = []
            print('Detecciones borradas.')