"""
Cliente de prueba que imita las comunicaciones del robot,
solicitando coordenadas al servidor.

Cada vez que se ejecuta, este cliente envía una solicitud al servidor,
y espera la respuesta.  Cuando llega la muestra en consola y termina.
La respuesta es una string, este cliente de pruebas no la analiza, la muestra tal como le llega.

Está pensado para ejecutarse en consola varias veces, cada vez que se quiere obtener una respuesta.

    .. code-block:: bash

        python test-client.py 

envía una solicitud a _localhost_, al puerto por defecto 65432, el mismo elegido para el servidor.
Útil cuando se ejecuta en la misma máquina que el servidor.


    .. code-block:: bash

        python test-client.py -i 192.168.0.101 -p 65432


permite elegir IP y puerto destino (los del servidor) 
y por lo tanto ejecutar este cliente desde otra máquina en la misma red.
"""

if __name__ == "__main__":
    import argparse
    import socket

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="server IP", default="localhost")
    parser.add_argument("-p", "--server_port", help="server port", type=int, default=65432)
    args = parser.parse_args()

    TCP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    TCP_socket.connect((args.server_ip, args.server_port))
    TCP_socket.sendall(b"Coordenadas, por favor")
    response = TCP_socket.recv(1024)  # Recibir la respuesta (hasta 1024 bytes), este comando es bloqueante
    print("Respuesta del servidor:", response.decode())