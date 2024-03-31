from bluepy.btle import Peripheral
import time
import keyboard
import threading

device_mac = 'DC:54:75:D8:4B:D9'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
char_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'

# Global flag to control the main loop
running = True
# Data to send, accessible and modifiable by both threads
data_to_send = ""


def capture_input():
    global running, data_to_send
    while running:
        user_input = input("Enter input or 'q' to quit: ")

        if user_input == 'q':
            running = False
        else:
            data_to_send = user_input

# Function to send data to ESP32


def send_data():
    global data_to_send
    with Peripheral(device_mac) as p:
        service = p.getServiceByUUID(service_uuid)
        char = service.getCharacteristics(char_uuid)[0]
        while running:
            value = char.read().decode()
            print(value)
            if data_to_send:
                char.write(data_to_send.encode('utf-8'))
                print(f"Sent: {data_to_send}")
                data_to_send = ""  # Reset data_to_send after sending
            time.sleep(1)  # Prevent this loop from hogging CPU

# Function to capture input in a non-blocking manner


# Start the input thread
input_thread = threading.Thread(target=capture_input)
input_thread.start()

# Start the send data function in the main thread
try:
    send_data()
finally:
    # Wait for the input thread to finish before exiting the program
    input_thread.join()
    print("Program ended.")
