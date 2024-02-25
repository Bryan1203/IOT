import socket
import picar_4wd as fc
import combine_multi as cm

HOST = "192.168.68.126" # IP address of your Raspberry PI
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
                print("data before: ",data)     
                print("data type: ",type(data))   
                client.sendall(data) # Echo back to client
                
                if b'W' in data:
                    print("W pressed!!!")
                    cm.goForward()
                if  b'S' in data:
                    cm.goBackward()
                if  b'A' in data:
                    cm.goLeft()
                if  b'D' in data:
                    cm.goRight()
                if b'U' in data:
                    client.sendall(fc.get_distance_at(0))


    except: 
        print("Closing socket")
        client.close()
        s.close()    