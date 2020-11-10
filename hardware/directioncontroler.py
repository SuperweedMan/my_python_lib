#%%
import time
from motorcontroler import MySerialForMotor
from pynput.keyboard import Controller,Key,Listener
#%%
class DirectionControl:
    def __init__(self, com_str, baud_rate):
        self.ser = MySerialForMotor(com_str, baud_rate)
        self.lis = Listener(on_press=self.on_press, on_release=self.on_release)
        self.current_pressed = []
        self.selection = ('up', 'down', 'right', 'left')
        self.current_direction = 'center'
        self.direction_map = {
            'up': (0.4, 0),
            'down': (-0.4, 0),
            'right': (0, -1.95),
            'left': (0, 1.95),
            'rightup': (0.2, -0.975),
            'leftup': (0.2, 0.975),
            'downleft': (-0.2, -0.975),
            'downright': (-0.2, 0.975)
        }

    def start_listen(self):
        self.lis.start()

    def join(self):
        self.lis.join()

    def on_press(self, key):
        if key.name in self.current_pressed:
            pass
        else:
            if key.name in self.selection:
                self.current_pressed.append(key.name)
                self.update_direction()

    def on_release(self, key):
        if key==Key.esc:  # 停止监听
            self.ser.close()
            print("exit keyboard monitor")
            return False
        if key.name in self.current_pressed:
            self.current_pressed.remove(key.name)
            self.update_direction()
    
    def update_direction(self):
        if len(self.current_pressed) == 0:
            self.ser.write_speed(0, 0)
        else:
            self.current_pressed.sort()
            pressed = ''.join(self.current_pressed)
            if pressed in self.direction_map:
                self.ser.write_speed(*self.direction_map[pressed])

if __name__ == '__main__':
    # ser = MySerialForMotor('COM3', 115200)
    dir = DirectionControl('/dev/ttyTHS2', 115200)
    dir.start_listen()
    # start_listen()
    print("start")
    dir.join()
