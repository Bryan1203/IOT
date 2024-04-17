from bluepy.btle import Peripheral
import time
import threading
import queue
import signal

# IDs
device_mac = 'DC:54:75:D8:4B:D9'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
char_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'

# globals
peripheral = Peripheral(device_mac)
service = peripheral.getServiceByUUID(service_uuid)
char = service.getCharacteristics(char_uuid)[0]


def send_message(mes):
    char.write(bytes(mes, "utf-8"))


def get_message():
    return char.read().decode()


def swap_directions():
    while True:
        print("Value: ", get_message())
        send_message("Right")
        time.sleep(3)

        print("Value: ", get_message())
        send_message("Left")
        time.sleep(3)


def terminal_input():
    while True:
        print("Value: ", get_message())
        user_input = input("Enter your message: ")
        send_message(user_input)


def signal_handler(sig, frame):
    send_message("Stop")
    peripheral.disconnect()
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    terminal_input()


if __name__ == "__main__":
    main()
