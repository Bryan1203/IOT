from bluepy.btle import Peripheral
import time
import threading
import queue

device_mac = 'DC:54:75:D8:4B:D9'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
char_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'


def swap_directions(peripheral):
    service = peripheral.getServiceByUUID(service_uuid)
    char = service.getCharacteristics(char_uuid)[0]

    while True:
        char.write(bytes("Right", "utf-8"))
        time.sleep(3)
        char.write(bytes("Left", "utf-8"))
        time.sleep(3)


def main():
    with Peripheral(device_mac) as p:
        swap_directions(p)


if __name__ == "__main__":
    main()
