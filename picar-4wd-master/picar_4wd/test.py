import picar_4wd as fc
import RPi.GPIO as GPIO
import time, math
import threading
from speed import Speed


def  move25():

     speed4 = Speed(25)
     speed4.start()
     # time.sleep(2)
     fc.forward(35)
     fc.stop()
     time.sleep(0.5)
     fc.turn_right(50)
     print('distance:',fc.us.get_distance())
'''    
    x = 0
     for i in range(15):
         time.sleep(0.1)
         speed = speed4()
         x += speed * 0.1
         print('distance:',fc.us.get_distance())
         print('%smm/s'%speed)
     print('%smm/s'%x)
     speed4.deinit()
     fc.turn_right(80)
   
     fc.stop()
''' 
def turnLeft():
     speed4 = Speed(25)
     speed4.start()
        # time.sleep(2)
     fc.turn_left(80)
     x = 0
     for i in range(15):
        time.sleep(0.1)
        speed = speed4()
        x += speed * 0.1
        print("%smm/s"%speed)
     print("%smm"%x)
     speed4.deinit()
     fc.stop()

def turnRight():
    speed4 = Speed(25)
    speed4.start()
        # time.sleep(2)
    fc.turn_right(80)
    x = 0
    for i in range(15):
        time.sleep(0.1)
        speed = speed4()
        x += speed * 0.1
        print("%smm/s"%speed)
    print("%smm"%x)
    speed4.deinit()
    fc.stop()     
        
#key = cv2.waitKey(1)
#if key == 27:
         #      break
     
if __name__ == "__main__":
   #move25()
	try:
		move25()
	except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
		print("stopped by User")
   #turnRight()
   #turnLeft()