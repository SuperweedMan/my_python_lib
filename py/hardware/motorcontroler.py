#%%
import serial
from positiondecode import PositionDecode
import struct


class MySerialForMotor:

    def __init__(self, com_str, baud_rate):
        assert isinstance(com_str, str) and isinstance(baud_rate, int)
        self.ser = serial.Serial(com_str, baud_rate, timeout=0.5)
        self.resolver = PositionDecode()
        if not self.ser.is_open:
            self.ser.open()

    def write_speed(self, forwardspeed, rollspeed):
        forwardspeed = float(forwardspeed)
        rollspeed = float(rollspeed)
        bytes_seq = bytes.fromhex('ff') + \
                        struct.pack('<f', forwardspeed) + \
                        struct.pack('<f', rollspeed) + \
                        bytes.fromhex('ff')
        self.ser.write(bytes_seq)

    def read_data(self):
        if self.ser.is_open:
            return self.resolver.run(self.ser.read(self.ser.in_waiting)[-100:])
        else:
            print('Please open serial first.')

    def __del__(self):
        self.close()

    def open(self):
        if not self.ser.is_open:
            self.ser.open()

    def close(self):
        if self.ser.is_open:
            self.ser.close()
#%%
if __name__ == '__main__':
    motor = MySerialForMotor('/dev/ttyTHS2', 115200)
    motor.write_speed(1.2, 1.2)
