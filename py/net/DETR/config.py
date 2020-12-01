import json
from easydict import EasyDict as edict
import os

folder_path, file_name = os.path.split(__file__)
setting_file_path = os.path.join(folder_path, 'setting.json')
if not os.path.exists(setting_file_path):
    raise ValueError("Can't find the setting.json file.")
with open(setting_file_path, 'r') as f:
    CONFIG = edict(json.load(f))

def Reflash():
    with open(setting_file_path, 'r') as f:
        CONFIG = edict(json.load(f))