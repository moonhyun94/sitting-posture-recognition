import video
import sensor
import time
from multiprocessing import Process, Event, Pipe

# Camera Window Size
WIDTH = 640
HEIGHT = 480
FPS = 30.0
CAMERA = 1 # 0

if __name__ == '__main__':
    # creating event object not set
    event = Event()
    event2 = Event()
    # video to main, main to video
    vtm_conn, mtv_conn = Pipe()
    stm_conn, mts_conn = Pipe()
    
    p1 = Process(name='video', target=video.capture_video, args=(vtm_conn, event, CAMERA, WIDTH, HEIGHT, FPS, 'A1', 'M1'))
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
    
    while True:
        if mtv_conn.recv() == 'q':
            event2.set()
            break