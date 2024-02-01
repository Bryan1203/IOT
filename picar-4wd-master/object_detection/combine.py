import threading
import object_detect
import mapping



t1 = threading.Thread(target=mapping.main)
t2 = threading.Thread(target=object_detect.main)

t1.start()
t2.start()

t1.join()
t2.join()
