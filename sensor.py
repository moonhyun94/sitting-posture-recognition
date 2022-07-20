import serial
from datetime import datetime
import cv2 as cv
BAUDRATE = 921600
MEHTOD = ['euler', 'quaternion']
# Camera Window Size
WIDTH = 640
HEIGHT = 480

def openSerial(port, baudrate=BAUDRATE, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=None, xonxoff=False, rtscts=False, dsrdtr=False):
    ser = serial.Serial()

    ser.port = port
    ser.baudrate = baudrate
    ser.bytesize = bytesize
    ser.parity = parity
    ser.stopbits = stopbits
    ser.timeout = timeout
    ser.xonxoff = xonxoff
    ser.rtscts = rtscts
    ser.dsrdtr = dsrdtr
    
    try:
        ser.open()
    except serial.SerialException as err:
        print('Serial Exception', err)
        exit()
    return ser

def read(ser, size=1, timeout=None, method='euler'):
    readed = ser.read_until()
    timestamp = str(datetime.now())
    readed = readed.decode()
    return readed, timestamp

def capture_sensor(conn, e, e2, action, subject):
    ser = openSerial('com3')
    print('Serial Port com3 open, Waiting for e ... ')
    conn.send('Ready')
    conn.close()
    
    ts = datetime.strftime(datetime.now(), "%Y-%m-%d %H-%M-%S")
    ac = action
    sb = subject
    output = open(f'data/sensor/{ts}_{ac}_{sb}.txt', mode='w')
    data = []
    
    try:
        while ser.is_open:
            readed, ts = read(ser)
            
            if e.is_set():
                output.write(ts+','+readed)
                data.append((readed, ts))
            if e2.is_set():
                break
                      
    finally:
        ser.close()
        output.close()
        