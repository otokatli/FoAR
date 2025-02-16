'''
Ethernet Force Torque Sensor Class
Author: Hongjie Fang
Ref: 
  [1] <Ethernet Axia F/T Sensor Manual>, available at https://www.ati-ia.com/zh-cn/app_content/documents/9610-05-Ethernet%20Axia.pdf
  [2] <OptoForce Installation and Operation Manual For Ethernet Converter>, available at https://manualzz.com/doc/26839390/ethernet---optoforce.
  [3] NetFT, Cameron Devine, Github, availabale at https://github.com/CameronDevine/NetFT
'''

import time
import socket
import struct
import threading
import numpy as np

from pynput import keyboard
from collections import deque
from multiprocessing import shared_memory


class EthernetFTSensor(object):
    '''
    Ethernet Force/Torque Sensor.
    '''
    def __init__(
        self, 
        ip = '192.168.1.1', 
        scale = (1000, 1000),
        shm_name = None,
        show_info = True,
        **kwargs
    ):
        '''
        Initialization.

        Parameters
        ----------
        ip: str, optional, default: '192.168.1.1', the IP address of the ethernet force/torque sensor;
        scale: tuple of (int, int), default: (1000, 1000), the scaling coefficient of the force and torque values;
        shm_name: str, optional, default: None, the shared memory name of the force/torque data reading from the sensor;
        show_info: bool, optional, default: True, whether to show the detailed on the screen.
        '''
        super(EthernetFTSensor, self).__init__()
        self.ip = ip
        self.port = 49152
        self.scale = np.array([scale[0]] * 3 + [scale[1]] * 3).astype(np.float32)
        self.mean = np.zeros((6, )).astype(np.float32)
        self.shm_name = shm_name
        self.show_info = show_info
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.connect((self.ip, self.port))
        self.is_streaming = False
    
    def send(self, command, data):
        '''
        Send a given command to the sensor.

        Parameters
        ----------
        command: uint16, required, refer to the value according to the command table;
        data: uint32, required, data according to the actual command.
        '''
        header = 0x1234
        msg = struct.pack('!HHI', header, command, data)
        self.socket.send(msg)
    
    def receive(self):
        '''
        Receive and unpack the response from the sensor.

        Returns
        -------
        np.array of float: The force and torque values received. The first three values are the forces recorded, and the last three are the measured torques.
        '''
        msg = self.socket.recv(1024)
        data = np.array(struct.unpack('!IIIiiiiii', msg)[3:]).astype(np.float32)
        self.data = data / self.scale - self.mean
        return self.data
    
    def getMeasurements(self, n):
        '''
        Request a given number of measurements from the sensor. Notice that this function requests a given number of samples from the sensor. These measurements must be received manually using the `receive` function.

        Parameters
        ----------
        n: uint32, required, sample count.
        '''
        self.send(0x0002, n)
    
    def getMeasurement(self):
        '''
        Get a single measurement from the sensor and return it. If the sensor is currently streaming, started by running `startStreaming`, then this function will simply return the most recently returned value.

        Returns
        -------
        np.array of float: The force and torque values received. The first three values are the forces recorded, and the last three are the measured torques.
        '''
        self.getMeasurements(n = 1)
        self.receive()
        return self.data

    def measurement(self):
        '''
        Get the most recent force/torque measurement.
        
        Returns
        -------
        np.array of float: The force and torque values received. The first three values are the forces recorded, and the last three are the measured torques.
        '''
        return self.data
    
    def lowPassMeasurement(self):
        '''
        Get the most recent force/torque measurement after low pass filter.
        
        Returns
        -------
        np.array of float: The force and torque values low pass filtered. The first three values are the forces recorded, and the last three are the measured torques.
        '''
        return self.low_pass_data

    def getForce(self):
        '''
        Get a single force measurement from the sensor. Request a single measurement from the sensor and return it.
		
        Returns
        -------
		np.array of float: The force values received.
		'''
        return self.getMeasurement()[:3]
    
    def force(self):
        '''
        Get the most recent force measurement.
		
        Returns
        -------
		np.array of float: The force values received.
        '''
        return self.measurement()[:3]
    
    def getTorque(self):
        '''
        Get a single torque measurement from the sensor. Request a single measurement from the sensor and return it.

		Returns
        -------
		np.array of float: The torque values received.
        '''
        return self.getMeasurement()[3:]
    
    def torque(self):
        '''
        Get the most recent torque measurement.

		Returns
        -------
		np.array of float: The torque values received.
        '''
        return self.measurement()[3:]
    
    def startStreaming(self, handler = True):
        '''
        Start streaming data continuously. This function commands the sensor to start sending data continuously. By default this also starts a new thread with a handler to save all data points coming in. These data points can still be accessed with `measurement`, `force`, and `torque`. This handler can also be disabled and measurements can be received manually using the `receive` function.

		Parameters
        ----------
		handler: bool, optional, default: True, if True start the handler which saves data to be used with `measurement`, `force`, and `torque`; if False the measurements must be received manually.
		'''
        self.getMeasurements(0)
        if handler:
            self.is_streaming = True
            self.thread = threading.Thread(target = self.receiveHandler)
            self.thread.setDaemon(True)
            self.thread.start()
    
    def receiveHandler(self):
        '''
        A handler to receive and store data.
        '''
        self._prepare_shm()
        if self.show_info:
            print('[Sensor] Start streaming ...')
        shm_flag = self.shm_name is not None
        while self.is_streaming:
            data = self.receive()
            if shm_flag:
                self.shm_buf[:] = data[:]

    def stopStreaming(self, delay_time = 0.1):
        '''
        Stop streaming data continuously. This function stops the sensor from streaming continuously as started using `startStreaming`.

        Parameters
        ----------
        delay_time: the delay time between close streaming thread and sending stop streaming signal to sensor.
		'''
        self.is_streaming = False
        time.sleep(delay_time)
        self.thread.join()
        self._close_shm()
        self.send(0x0000, 0)

    def zero(self):
        '''
        Remove the mean found with `tare` to start receiving raw sensor values.
        '''
        self.mean = np.zeros((6, )).astype(np.float32)
    
    def tare(self, n = 100):
        '''
        Tare the sensor. This function takes a given number of readings from the sensor and averages them. This mean is then stored and subtracted from all future measurements.
		
        Parameters
        ----------
		n: int, optional, default: 10, the number of samples to use in the mean.

		Returns
        -------
        np.array of float: The mean values calculated.
        '''
        mean = np.zeros((6, )).astype(np.float32)
        self.getMeasurements(n = n)
        for _ in range(n):
            self.receive()
            mean += self.measurement() / float(n)
        self.mean = mean
        return self.mean

    def _prepare_shm(self):
        '''
        Prepare shared memory objects.
        '''
        if self.shm_name is not None:
            measurement = self.receive()
            self.shm = shared_memory.SharedMemory(name = self.shm_name, create = True, size = measurement.nbytes)
            self.shm_buf = np.ndarray(measurement.shape, dtype = measurement.dtype, buffer = self.shm.buf)
    
    def _close_shm(self):
        '''
        Close shared memory objects.
        '''
        if self.shm_name is not None:
            self.shm.close()
            self.shm.unlink()
        print('[Sensor] Closed.')


class OptoForceFTSensor(EthernetFTSensor):
    '''
    OptoForce Force/Torque Sensor.
    '''
    def __init__(
        self, 
        ip = '192.168.3.22', 
        scale = (10000, 100000),
        shm_name = 'force_torque',
        show_info = True,
        
        **kwargs
    ):
        super(OptoForceFTSensor, self).__init__(ip, scale, shm_name, show_info, **kwargs)
    
    def run(self, tare = True):
        '''
        Sample streaming process, press 'q' to stop streaming.

        Parameters
        ----------
        tare: bool, optional, default: True, whether to tare before streaming.
        '''
        self.streaming(tare = tare)
        self.listener = keyboard.Listener(on_press = self._on_press, on_release = self._on_release)
        self.listener.start()
        while self.is_streaming:
            pass
        self.stop_streaming()
        self.listener.join()
    
    def streaming(self, tare = True):
        '''
        Start streaming process.

        Parameters
        ----------
        tare: bool, optional, default: True, whether to tare before streaming.
        '''
        if tare:
            self.tare()
        else:
            self.zero()
        self.startStreaming(handler = True)

    def stop_streaming(self):
        '''
        Stop streaming process.
        '''
        self.stopStreaming(delay_time = 0)
    
    def _on_press(self, key):
        try:
            if key.char == 'q':
                self.is_streaming = False
                return False
        except AttributeError:
            pass
    
    def _on_release(self, key):
        pass


class OptoForceFTSensorWithHistory(OptoForceFTSensor):
    def __init__(
        self, 
        history_size = 100,  
        ip = '192.168.3.22', 
        scale = (10000, 100000),
        shm_name = 'force_torque',
        show_info = True,
        **kwargs
    ):
        super(OptoForceFTSensorWithHistory, self).__init__(ip, scale, shm_name, show_info, **kwargs)
        
        initial_data = np.zeros(6)
        self.history = deque([initial_data.copy() for _ in range(history_size)], maxlen = history_size)
    
    def receiveHandler(self):
        '''
        A handler to receive and store data.
        '''
        self._prepare_shm()
        if self.show_info:
            print('[Sensor] Start streaming ...')
        shm_flag = self.shm_name is not None
        while self.is_streaming:
            data = self.receive()      
            if shm_flag:
                self.shm_buf[:] = data[:]
            
            self.history.append(data)
    
    def getHistory(self):
        '''
        get history of force and torque value

        Returns
        -------
        np.array of float: history of force and torque value
        '''
        return np.array(self.history)

    def getHistoryfreq(self, freq = 100):
        '''
        get history of force and torque value with a given frequency

        Parameters
        ----------
        freq: int, optional, default: 100, the frequency of the history.
        
        Returns
        -------
        np.array of float: history of force and torque value with a given frequency
        '''
        step = int(100.0 / freq)
        return np.array(self.history)[::step]


    def stop_streaming(self):
        '''
        Stop streaming process.
        '''
        self.stopStreaming(delay_time = 0)
    
    def _close_shm(self):
        '''
        Close shared memory objects.
        '''
        if self.shm_name is not None:
            self.shm.close()
            self.shm.unlink()
        print('[Sensor] Closed.')