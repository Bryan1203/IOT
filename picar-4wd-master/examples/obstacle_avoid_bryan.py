import time
import picar_4wd as fc

speed = 30

def main():
    while True:
        scan_list = fc.scan_step(35)
        if not scan_list:
            continue

        tmp = scan_list[3:7]
        print(tmp)
        if len(tmp) == 2:
            left_side = tmp[0]
            right_side = tmp[1]
        elif len(tmp) == 4:
            left_side = tmp[0]+tmp[1]
            right_side = tmp[2]+tmp[3]

        sum = left_side +right_side 
        if tmp == [0,0,0,0] or sum <=1:
            fc.backward(speed)
            #time.sleep(0.75)
        elif left_side > right_side:
            fc.turn_left(speed)
        elif right_side > left_side:
            fc.turn_right(speed)
        else:
            fc.forward(speed)

if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()
