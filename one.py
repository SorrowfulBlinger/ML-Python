import numpy as np
import threading
import time


class AsyncWorker(threading.Thread):
    __lock = threading.Lock()

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        for i in range(1, 5, 1):
            AsyncWorker.__lock.acquire()
            print 'async work by ' + threading.current_thread().__str__()
            time.sleep(.5)
            AsyncWorker.__lock.release()


def asyncMethod(input_param):
    for i in range(1, 5, 1):
        print threading.current_thread(),
        print input_param,
        print  np.random.normal(50, i, 10)
        time.sleep(2)


print (lambda x: x*x)(2)

t1 = threading.Thread(target=asyncMethod, args=['Thread 1'])
t2 = threading.Thread(target=asyncMethod, args=['Thread 2'])

a1 = AsyncWorker()
a2 = AsyncWorker()

a1.start()
a2.start()

a2.join()
a1.join()


# t1.start()
# t2.start()

# t1.join()
# t2.join()
