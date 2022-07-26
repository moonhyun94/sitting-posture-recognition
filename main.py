import video
import sensor
import time
import keyboard
from multiprocessing import Process, Event, Pipe


BAUDRATE = 921600
MEHTOD = ['euler', 'quaternion']
# Camera Window Size
WIDTH = 640
HEIGHT = 480

if __name__ == '__main__':
    # creating event object not set
    event = Event()
    event2 = Event()
    # video to main, main to video
    vtm_conn, mtv_conn = Pipe()
    stm_conn, mts_conn = Pipe()
    
    p1 = Process(name='video', target=video.capture_video, args=(vtm_conn, event, 1, WIDTH, HEIGHT, 30.0, 'A1', 'M1'))
    p2 = Process(name='sensor', target=sensor.capture_sensor, args=(stm_conn, event, event2, 'A1', 'M1'))
    
    p1.start()
    p2.start()
    
    print('main: waiting response before calling event.set()')
    
    while True:
        if mtv_conn.recv() and mts_conn.recv() == 'Ready':
            break
        
    print('main: setting event in 5sec ...')
    time.sleep(5)
    
    event.set()
    
    # blocked until recv
    while True:
        if mtv_conn.poll():
            msg = mtv_conn.recv()
            if msg == 'p':
                event.clear()
            if msg == 'r':
                event.set()
            if msg == 'q':
                event2.set()
                break