import picar_4wd as fc
import time

def test_speed(): 
	speed4 = fc.Speed(4)
	speed25 = fc.Speed(25)
	speed4.start()
	speed25.start()
    # time.sleep(2)
	fc.forward(60)
	x = 0
	#for i in range(20):
		#time.sleep(0.1)
		#speed = speed4()
		#x += speed * 0.1
		#print("%smm/s"%speed)
	distance = 25

	while x <= distance:
		fc.turn_right(1)
		x += (speed4()+speed25()) * 0.1
		time.sleep(0.1)
		
	
	print("%smm"%x)
	speed4.deinit()
	speed25.deinit()
	fc.stop()
	
if __name__ == '__main__':
    #fc.start_speed_thread()
    test_speed()
