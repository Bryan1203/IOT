# Import SDK packages
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import json
import pandas as pd
import numpy as np
import boto3


#TODO 1: modify the following parameters
#Starting and end index, modify this
device_st = 0
device_end = 1

region_name = 'us-east-2'
#Path to the dataset, modify this
data_path = r"data2\vehicle1.csv"


#Path to your certificates, modify this

certificate_formatter = r"C:\Users\Bryan\Desktop\CS437\IOT\lab4\Car\f74fc04c99211f7ffb5f1cc3a2e0043289669a1d5c9fe19a81dc5d7f7b28e2cc-certificate.pem.crt"
key_formatter = r"C:\Users\Bryan\Desktop\CS437\IOT\lab4\Car\f74fc04c99211f7ffb5f1cc3a2e0043289669a1d5c9fe19a81dc5d7f7b28e2cc-private.pem.key"


# Create an IoT client
iot_client = boto3.client("iot", region_name=region_name)

# Get the list of things
response = iot_client.list_things()
things = response["things"]

class MQTTClient:
    def __init__(self, device_id, cert, key):
        # For certificate based connection
        self.device_id = str(device_id)
        self.state = 0
        self.client = AWSIoTMQTTClient(self.device_id)
        #TODO 2: modify your broker address
        self.client.configureEndpoint("a2l7dk2u7na085-ats.iot.us-east-2.amazonaws.com", 8883)
        self.client.configureCredentials("AmazonRootCA1.pem", key, cert)
        self.client.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
        self.client.configureDrainingFrequency(2)  # Draining: 2 Hz
        self.client.configureConnectDisconnectTimeout(10)  # 10 sec
        self.client.configureMQTTOperationTimeout(5)  # 5 sec
        self.client.onMessage = self.customOnMessage
        

    def customOnMessage(self,message):
        #TODO3: fill in the function to show your received message
        print("client {} received payload {} from topic {}".format(self.device_id, message.payload, message.topic))


    # Suback callback
    def customSubackCallback(self,mid, data):
        #You don't need to write anything here
        pass


    # Puback callback
    def customPubackCallback(self,mid):
        #You don't need to write anything here
        pass


    def publish(self, Payload="payload"):
        #TODO4: fill in this function for your publish
        self.client.subscribeAsync("iot/Vehicle", 0, ackCallback=self.customSubackCallback)
        
        self.client.publishAsync("myTopic", Payload, 0, ackCallback=self.customPubackCallback)



print("Loading vehicle data...")
data = []
for i in range(5):
    a = pd.read_csv(data_path.format(i))
    data.append(a)

print("Initializing MQTTClients...")
clients = []
for thing in things:
    device_id = thing["thingName"]
    client = MQTTClient(device_id,certificate_formatter.format(device_id,device_id) ,key_formatter.format(device_id,device_id))
    client.client.connect()
    clients.append(client)
 
counter = 0
while True:
    print("send now?")
    x = input()
    if x == "s":
        
        #json_str = (data[0].loc[counter]).to_json(orient='records')
        result = (data[0].iloc[counter].to_dict())
        json_output = json.dumps(result, indent=4)
        print(result)
        clients[0].publish(json_output)
        counter+=1
            


    elif x == "d":
        for c in clients:
            c.client.disconnect()
        print("All devices disconnected")
        exit()
    else:
        print("wrong key pressed")

    time.sleep(3)