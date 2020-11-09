import transitions
import sys
import struct
from transitions import State, Machine
sys.setrecursionlimit(1000)

class PositionDecode:
    def __init__(self):
        self.head_of_frame = bytes.fromhex('fe ef')  # 转为bytes
        self.end_of_frame = bytes.fromhex('aa aa')  
        self.return_list = []
        states = [
            State(name='find_frame_head', on_enter=['find_head_of_frame']),
            State(name='find_frame_taril', on_enter=['find_taril_of_frame']),
            State(name='verify', on_enter=['verify']),
            State(name='analysis', on_enter=['analysis'])
        ]
        transition_map = [
            { 'trigger': 'input_data',   'source': 'find_frame_taril', 'dest': 'find_frame_taril' },
            { 'trigger': 'find_taril', 'source': 'find_frame_taril', 'dest': 'find_frame_head' },
            { 'trigger': 'no_head', 'source': 'find_frame_head', 'dest': 'find_frame_taril' },
            { 'trigger': 'find_head', 'source': 'find_frame_head', 'dest': 'verify' },
            { 'trigger': 'vpass', 'source': 'verify', 'dest': 'analysis' },
            { 'trigger': 'vreject', 'source': 'verify', 'dest': 'find_frame_taril' },
            { 'trigger': 'new_loop', 'source': 'analysis', 'dest': 'find_frame_taril' },            
        ]
        self.machine = transitions.Machine(self, states=states, transitions=transition_map, initial='find_frame_taril')
        # self.cache = b''
        self.raw_cache = b''

    def find_taril_of_frame(self, bytes_seq):
        assert isinstance(bytes_seq, bytes)
        cache, taril, self.raw_cache = (self.raw_cache + bytes_seq).partition(self.end_of_frame)  # 分割出可能存在frame的缓存跟剩余的数据
        if not taril == b'':  # 找到了
            self.find_taril(cache)
        else:
            if taril == b'' and self.raw_cache == b'':
                self.raw_cache = cache
    
    def find_head_of_frame(self, bytes_seq):
        assert isinstance(bytes_seq, bytes)
        if self.head_of_frame in bytes_seq:
            self.find_head(
                bytes_seq[bytes_seq.find(self.head_of_frame):])
        else:
            self.no_head(b'')

    def verify(self, bytes_seq):
        assert isinstance(bytes_seq, bytes)
        num = bytes_seq[len(self.head_of_frame)]
        if len(bytes_seq) - len(self.head_of_frame) - 1 == num:  # 校验通过
            self.vpass(bytes_seq[len(self.head_of_frame) + 1:])
        else:
            self.vreject(b'')

    def analysis(self, bytes_seq):
        assert isinstance(bytes_seq, bytes)
        position_dict = {'x_position': struct.unpack('<f', bytes_seq[:4])[0],
                        'y_position': struct.unpack('<f', bytes_seq[4:])[0]}
        self.return_list.append(position_dict)
        self.new_loop(b'')

    def run(self, bytes_seq):
        assert isinstance(bytes_seq, bytes)
        self.return_list = []
        self.input_data(bytes_seq)
        return self.return_list

