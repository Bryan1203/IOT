from bluepy.btle import Peripheral, BTLEException, BTLEDisconnectError
import time
import threading
import queue
import signal

# IDs
device_mac = 'DC:54:75:D8:4B:D9'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
char_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'


def connect_to_peripheral():
    while True:
        # print("hello")
        try:
            # print("hoi")
            peripheral = Peripheral(device_mac)
            print("Connected to peripheral.")
            return peripheral
        except (BTLEException, BTLEDisconnectError) as e:
            print("Failed to connect to peripheral. Retrying in 2 seconds...")
            time.sleep(2)


def reconnect_peripheral():
    # global peripheral, service, char
    global peripheral, service, char
    peripheral.disconnect()
    peripheral = connect_to_peripheral()
    service = peripheral.getServiceByUUID(service_uuid)
    char = service.getCharacteristics(char_uuid)[0]


# globals
peripheral = connect_to_peripheral()
service = peripheral.getServiceByUUID(service_uuid)
char = service.getCharacteristics(char_uuid)[0]


def send_message(mes):
    try:
        char.write(bytes(mes, "utf-8"))
    except (BTLEException, BTLEDisconnectError):
        print("Connection lost. Attempting to reconnect...")
        reconnect_peripheral()
        send_message(mes)


def get_message():
    try:
        return char.read().decode()
    except (BTLEException, BTLEDisconnectError):
        print("Connection lost. Attempting to reconnect...")
        reconnect_peripheral()
        return get_message()


def swap_directions():
    while True:
        print("Value: ", get_message())
        send_message("Right")
        time.sleep(3)

        print("Value: ", get_message())
        send_message("Left")
        time.sleep(3)


def terminal_input():
    # print("hi")
    while True:
        print("Value: ", get_message())
        user_input = input("Enter your message: ")
        send_message(user_input)


def signal_handler(sig, frame):
    send_message("Disconnected")
    peripheral.disconnect()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    terminal_input()


if __name__ == "__main__":
    main()
