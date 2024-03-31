from bluepy.btle import Peripheral
import time
import keyboard
import threading

device_mac = 'DC:54:75:D8:4B:D9'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
char_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'

# Global flag to control the main loop
running = True


def capture_input():
    global running
    while running:
        user_input = input("Enter input or 'q' to quit: ")

        if user_input == 'q':
            running = False
        else:
            with Peripheral(device_mac) as p:
                service = p.getServiceByUUID(service_uuid)
                char = service.getCharacteristics(char_uuid)[0]

                char.write(user_input.encode('utf-8'))
                print(f"Sent: {user_input}")

# Function to send data to ESP32


def read_data():
    global running
    with Peripheral(device_mac) as p:
        service = p.getServiceByUUID(service_uuid)
        char = service.getCharacteristics(char_uuid)[0]
        while running:
            value = char.read().decode()
            print(value)

            time.sleep(1)  # Prevent this loop from hogging CPU

# Function to capture input in a non-blocking manner


# Start the input thread
input_thread = threading.Thread(target=capture_input)
input_thread.start()

# Start the send data function in the main thread
try:
    read_data()
finally:
    # Wait for the input thread to finish before exiting the program
    input_thread.join()
    print("Program ended.")
