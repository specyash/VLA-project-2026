# save as test_connect.py and run it
from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.152', do_not_open=False)
print("State:", arm.get_state())
print("Position:", arm.get_position())
arm.disconnect()