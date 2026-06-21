import time
import sys
import os

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.robot.xarm_robot import XArmRobotAdapter

def main():
    print("=== Testing Real xArm Custom Gripper (TGPIO Pin 0) ===")
    
    # Initialize the adapter with dry_run=False (real hardware) and use_voice=False for simplicity
    try:
        robot = XArmRobotAdapter(dry_run=False, use_voice=False)
    except Exception as e:
        print(f"Failed to connect or initialize: {e}")
        return

    try:
        print("\n--- Command: Open Gripper (TGPIO 0 = 0) ---")
        robot.open_gripper(description="opening gripper")
        time.sleep(2)

        print("\n--- Command: Close Gripper (TGPIO 0 = 1) ---")
        robot.close_gripper(description="closing gripper")
        time.sleep(2)

        print("\n--- Command: Open Gripper (TGPIO 0 = 0) ---")
        robot.open_gripper(description="opening gripper again")
        time.sleep(1)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()
