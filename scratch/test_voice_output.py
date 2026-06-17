import time
from src.robot.xarm_robot import XArmRobotAdapter

def main():
    print("=== Testing xArm Voice Dictation (Dry Run Mode) ===")
    
    # Initialize the adapter in dry run mode (with voice active)
    robot = XArmRobotAdapter(dry_run=True, use_voice=True)
    
    # Sequence of commands to execute (will trigger TTS)
    print("\nExecuting command sequence...")
    
    print("\n--- Command: Go Home ---")
    robot.home()
    time.sleep(0.5)

    print("\n--- Command: Open Gripper ---")
    robot.open_gripper(description="opening gripper")
    time.sleep(0.5)

    print("\n--- Command: Move to Pick Location ---")
    robot.move_to_pose(x=150.0, y=-200.0, z=120.0, description="moving to pick up object")
    time.sleep(0.5)

    print("\n--- Command: Close Gripper ---")
    robot.close_gripper(description="grasping object")
    time.sleep(0.5)

    print("\n--- Command: Lift Object ---")
    robot.move_to_pose(x=150.0, y=-200.0, z=200.0, description="lifting object")
    time.sleep(0.5)

    print("\n--- Command: Move to Bin ---")
    robot.move_to_pose(x=-100.0, y=-300.0, z=200.0, description="moving to bin")
    time.sleep(0.5)

    print("\n--- Command: Place Object ---")
    robot.move_to_pose(x=-100.0, y=-300.0, z=130.0, description="moving to drop object")
    time.sleep(0.5)

    print("\n--- Command: Open Gripper ---")
    robot.open_gripper(description="releasing object")
    time.sleep(0.5)

    print("\n--- Command: Return Home ---")
    robot.home()
    time.sleep(0.5)

    # Wait for the background voice queue to play everything out completely
    if robot.voice and robot.voice.active:
        print("\nWaiting for TTS voice output to finish playing...")
        robot.voice.queue.join()
        # Allow extra time for the last audio clip to finish playing before exiting
        time.sleep(2.0)
        
    print("\nSequence finished.")

if __name__ == "__main__":
    main()
