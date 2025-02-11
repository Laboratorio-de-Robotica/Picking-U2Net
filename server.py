'''
TCP server listening for incoming queries from robot, asking por coordinates for picking.
In combination with computer vision system locating objects with U2Net.
And a calibration routine for extrinsic camera calibration and homography matrix computation.

En desarrollo, no funcional aún.
'''

import socket
import threading
import time
import cv2 as cv
import numpy as np
from pick import PickU2Net
from lib.extrinsics import ExtrinsicCalibrator

print('''
Teclas:
      C: Calibrar extrínsecamente (computa la homografía)
      D: Detectar la posición de la pieza
      ESC: Salir del bucle
      Ctrl+C: Detener el servidor y salir del programa
''')

# Coordenadas de picking que se informarán al robot
x = 0.0
y = 0.0
angle = 0.0
aperture = 0.0
object_detected = False


# Get this server IP address
def get_my_ip_address(test="8.8.8.8"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    myIP = ""
    try:
        sock.connect((test, 80))
        myIP = sock.getsockname()[0]
    finally:
        sock.close()
        return myIP

'''
Función que ejecuta el servidor TCP/IP en un hilo aparte

Protocolo:
- El servidor recibe una consulta en formato de string, que no se analiza
- La string de respuesta tiene el fomato requerido por asciiToFloat, con esta información:
    "({x}, {y}, {angle} ,{aperture}, {1.0 if object_detected else 0.0})\n"
'''
def start_server():
    global object_detected

    # Crear un socket TCP/IP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print(f"Servidor escuchando en {HOST}:{PORT}...")

        # Bucle infinito para aceptar múltiples conexiones
        while True:
            # Usamos un timeout para evitar bloqueo completo en accept
            s.settimeout(1.0)
            try:
                # accept() bloquea hasta que recibe una consulta
                # addr contiene la ip de la máquina que realiza la consulta
                conn, addr = s.accept()
            except socket.timeout:
                # Si no hay conexión, seguimos esperando
                continue

            with conn:
                # Conexión establecida, recibir la consulta del cliente, una string
                data = conn.recv(1024)
                consulta = data.decode()
                print(f"Consulta: {consulta}")

                # Enviar las variables al cliente con el fomato requerido por asciiToFloat
                response = f"({x}, {y}, {angle} ,{aperture}, {1.0 if object_detected else 0.0})\n"
                conn.sendall(response.encode())
                print(f"Respuesta: {response}")

                # Reiniciar la variable de objeto detectado
                object_detected = False

                # Respuesta enviada. Conexión cerrada. Esperando nueva conexión...




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
            extrinsicCalibrator.calibrate(im)

        elif key == 'd':
            pickU2Net.detect(im)

except KeyboardInterrupt:
    print("Programa interrumpido por el usuario.")