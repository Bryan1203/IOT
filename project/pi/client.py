from bluepy.btle import Peripheral
import time
import threading
import queue

device_mac = 'DC:54:75:D8:4B:D9'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
char_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'

# Global flag to control the main loop
running = True

# Queue for thread-safe communication
input_queue = queue.Queue()


def capture_input():
    global running

    while running:
        user_input = input("Enter input or 'q' to quit: ")
        if user_input == 'q':
            running = False
        else:
            input_queue.put(user_input)
            print(f"Queued for sending: {user_input}")


def send_data(peripheral):
    global running
    service = peripheral.getServiceByUUID(service_uuid)
    char = service.getCharacteristics(char_uuid)[0]
    while running:
        if not input_queue.empty():
            user_input = input_queue.get()
            char.write(user_input.encode('utf-8'))
            print(f"Sent: {user_input}")


def read_data(peripheral):
    global running
    service = peripheral.getServiceByUUID(service_uuid)
    char = service.getCharacteristics(char_uuid)[0]
    while running:
        value = char.read().decode()
        print(value)
        time.sleep(1)  # Prevent this loop from hogging CPU


def main():
    global running
    with Peripheral(device_mac) as p:
        # Start the input and send data threads
        input_thread = threading.Thread(target=capture_input)
        send_thread = threading.Thread(target=send_data, args=(p,))
        input_thread.start()
        send_thread.start()

        try:
            read_data(p)
        finally:
            running = False
            input_thread.join()
            send_thread.join()
            print("Program ended.")


if __name__ == "__main__":
    main()
