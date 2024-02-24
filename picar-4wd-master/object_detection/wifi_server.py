import socket
import picar_4wd as fc
import combine_multi as cm

HOST = "192.168.3.49" # IP address of your Raspberry PI
PORT = 65432          # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    try:
        while 1:
            client, clientInfo = s.accept()
            print("server recv from: ", clientInfo)
            data = client.recv(1024)      # receive 1024 Bytes of message in binary format
            if data != b"":
                print(data)     
                client.sendall(data) # Echo back to client
                if data == "W":
                    cm.goForward()
                if data == "S":
                    cm.goBackward()
                if data == "A":
                    cm.goLeft()
                if data == "D":
                    cm.goRight()

    except: 
        print("Closing socket")
        client.close()
        s.close()    