from gpiozero import Servo
from time import sleep

# 你们现在的接线：BCM 编号
VALVE_GPIO = 20   # 电磁阀
PUMP_GPIO = 21    # 气泵

# 尽量模拟 Arduino Servo 默认信号：
# write(0)   -> 544 us
# write(180) -> 2400 us
# frame 20ms -> 50Hz
MIN_PULSE = 544 / 1_000_000
MAX_PULSE = 2400 / 1_000_000
FRAME = 20 / 1000

valve = Servo(
    VALVE_GPIO,
    min_pulse_width=MIN_PULSE,
    max_pulse_width=MAX_PULSE,
    frame_width=FRAME,
    initial_value=-1,
)

pump = Servo(
    PUMP_GPIO,
    min_pulse_width=MIN_PULSE,
    max_pulse_width=MAX_PULSE,
    frame_width=FRAME,
    initial_value=-1,
)

def write_0(device):
    # 对应 Arduino servo.write(0)
    device.value = -1

def write_180(device):
    # 对应 Arduino servo.write(180)
    device.value = 1

try:
    for _ in range(10):
        # 气泵工作 1 秒：电磁阀关，气泵开
        write_0(valve)
        write_180(pump)
        sleep(1.0)

        # 电磁阀工作 0.8 秒：气泵关，电磁阀开
        write_180(valve)
        write_0(pump)
        sleep(0.8)

        # 全部关闭
        write_0(valve)
        write_0(pump)
        sleep(0.2)

finally:
    # 程序结束前也强制关
    write_0(valve)
    write_0(pump)
    sleep(0.5)