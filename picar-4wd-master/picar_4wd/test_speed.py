from speed import Speed
import picar_4wd as fc
import time

def test_speed(): 
	speed4 = Speed(25)
	speed4.start()
    # time.sleep(2)
	fc.forward(35)
	x = 0
	for i in range(20):
		time.sleep(0.1)
		speed = speed4()
		x += speed * 0.1
		print("%smm/s"%speed)
	print("%smm"%x)
	speed4.deinit()
	fc.stop()
	
if __name__ == '__main__':
    #fc.start_speed_thread()
    test_speed()
