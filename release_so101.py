from src.robot.so101_mover import SO101SmoothMover
mover = SO101SmoothMover(port='/dev/ttyACM0')
mover.connect()
mover.release()
WAITING_POSE = {
    "elbow_flex.pos": 9.714285714285714,
    "gripper.pos": 6.731436502428869,
    "shoulder_lift.pos": -81.93406593406593,
    "shoulder_pan.pos": 5.450549450549451,
    "wrist_flex.pos": 76.08791208791209,
    "wrist_roll.pos": -0.04395604395604396,
}
mover.move_to(WAITING_POSE)
mover.disconnect()
print("SO101SmoothMover torque disabled.")

