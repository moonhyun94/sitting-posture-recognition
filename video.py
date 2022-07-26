import cv2 as cv
from datetime import datetime


def open_cam(cam_num, W, H):
    cap = cv.VideoCapture(cam_num)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, H)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()
    return cap

def save(outpath, W, H, FPS):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(outpath, fourcc, FPS, (W, H))
    return writer

def capture_video(conn, e, cam_num, WIDTH, HEIGHT, FPS, action, subject):
    
    cap = open_cam(cam_num, WIDTH, HEIGHT)
    ts, ac, sb = datetime.strftime(datetime.now(), "%Y-%m-%d %H-%M-%S"), action, subject
    
    video = save(f'data/video/{ts}_{ac}_{sb}.mp4', WIDTH, HEIGHT, FPS)
    timeout = open(f'data/video/{ts}_{ac}_{sb}_timestamp.txt', mode='w')
    
    print('video capture open, Waiting for e ... ')
    
    vid = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            timestamp = str(datetime.now())+'\n'
            
            if e.is_set():
                # cv.putText(frame, timestamp+str(cap.get(cv.CAP_PROP_FPS)), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
                # vid.append((timestamp, frame))
                timeout.write(timestamp)
                video.write(frame)
            else:
                cv.putText(frame, 'Not Recording', (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
                conn.send('Ready')
                
            cv.imshow('frame', frame)
            
            # pause resume quit
            key = cv.waitKey(1)
            
            if key == ord('p'):
                conn.send('p')
            if key == ord('r'):
                conn.send('r')
            if key == ord('q'):
                conn.send('q')
                break
    finally:
        cap.release()
        video.release()
        timeout.close()
        cv.destroyAllWindows()