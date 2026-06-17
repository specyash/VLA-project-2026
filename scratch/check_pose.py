from xarm.wrapper import XArmAPI

def main():
    ip = '192.168.1.152'
    print(f"Connecting to xArm at {ip}...")
    arm = XArmAPI(ip)
    
    code, pos = arm.get_position()
    if code == 0:
        print("\n--- Current Robot Pose ---")
        print(f"X:     {pos[0]:.2f} mm")
        print(f"Y:     {pos[1]:.2f} mm")
        print(f"Z:     {pos[2]:.2f} mm")
        print(f"Roll:  {pos[3]:.2f}°")
        print(f"Pitch: {pos[4]:.2f}°")
        print(f"Yaw:   {pos[5]:.2f}°")
        print("--------------------------\n")
    else:
        print(f"Failed to get position, code: {code}")
        
    arm.disconnect()

if __name__ == "__main__":
    main()
