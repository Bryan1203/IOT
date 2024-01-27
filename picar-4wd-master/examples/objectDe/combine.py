import threading
import object_detect
import keyboard_control

t1 = threading.Thread(target=object_detect.main())
t2 = threading.Thread(target=keyboard_control.main())

t1.start()
t2.start()