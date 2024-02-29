import socket
import picar_4wd as fc
import combine_multi as cm

HOST = "192.168.68.81" # IP address of your Raspberry PI
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
                #client.sendall(fc.get_distance_at(0)) # Echo back to client
                
                if b'W' in data:
                    print("W pressed!!!")
                    cm.goForward()
                    client.sendall(str(cm.getOrientation()).encode())
                if  b'S' in data:
                    cm.goBackward()
                    client.sendall(str(cm.getOrientation()).encode())
                if  b'A' in data:
                    cm.goLeft()
                    client.sendall(str(cm.getOrientation()).encode())
                if  b'D' in data:
                    cm.goRight()
                    client.sendall(str(cm.getOrientation()).encode())
                if b'U' in data:
                    USdata = fc.get_distance_at(0)
                    print(type(USdata))
                    #print(type(struct.pack('f',USdata)))
                    client.sendall(str(USdata).encode())


    except: 
        print("Closing socket")
        client.close()
        s.close()    
