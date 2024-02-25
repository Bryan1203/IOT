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
                client.sendall(data) # Echo back to client
                clean_data = data.replace('\r\n','')
                clean_data = clean_data.replace('b','')
                print("data after: ",clean_data)
                if clean_data == "W":
                    print("W pressed!!!")
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