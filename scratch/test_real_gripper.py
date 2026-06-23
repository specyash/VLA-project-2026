import time
import sys
import os

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.robot.xarm_robot import XArmRobotAdapter

def main():
    print("=== Testing Real xArm Gripper (Open then Close) ===")
    
    # Initialize the adapter with dry_run=False (real hardware) and use_voice=True
    try:
        robot = XArmRobotAdapter(dry_run=False, use_voice=True)
    except Exception as e:
        print(f"Failed to connect or initialize: {e}")
        return

    try:
        print("\n--- Command: Open Gripper ---")
        robot.open_gripper(description="opening gripper")
        time.sleep(3.0)

        print("\n--- Command: Close Gripper ---")
        robot.close_gripper(description="closing gripper")
        time.sleep(3.0)

        # Wait for any background TTS speech to finish
        if robot.voice and robot.voice.active:
            print("\nWaiting for TTS voice output to finish playing...")
            robot.voice.queue.join()
            time.sleep(2.0)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()
