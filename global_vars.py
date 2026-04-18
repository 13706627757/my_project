recyclable_signal = 0
foodScrap_signal = 0
hazardous_signal = 0
others_signal = 0
# 用于防止掉下的物品被识别为满载
recyclable_signal1 = 0
foodScrap_signal1 = 0
hazardous_signal1 = 0
others_signal1 = 0
port0 = '/dev/ttyUSB0'
port1 = '/dev/ttyUSB1'
# 用于终止video和引脚检测线程
video_stop = False
gpio_stop = False
singal0_list = ['T','t', 'y', 'w', 'K', 'C', 'Q', 'Y']
# singal0_list = ['y']
recyclable=["cans",  "bottles","papercut"]
foodScrap=["carrots", "potato", "turnip"]
hazardous= ["battery", 'medicine_box','capsule','ointment','inner_packaging','medicine_plate']
others=["china", "cobble","brick"]
is_infering=False
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
update_list=[ 2 for _ in range(0,100)]