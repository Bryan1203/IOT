import threading
import object_detect
import keyboard_control

t2 = threading.Thread(target=object_detect.main())
t1 = threading.Thread(target=keyboard_control.Keyborad_control())

t1.start()
t2.start()