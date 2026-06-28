# Agent Notes

## Hardware Safety Rules

- Any move whose destination is a board pose-map position, including board
  cells and board corners, must be routed through `robot.waiting_pose` first.
- Never move directly from one board position to another board position.
- Never move directly from an arbitrary low board position to a board target.
- Safe board-target path shape is:
  `current -> waiting_pose -> board_target`.
- When visiting multiple board targets, the safe path shape is:
  `target_a -> waiting_pose -> target_b -> waiting_pose -> target_c`.
- After a board-target move finishes, return to `waiting_pose` before camera
  sync, human turns, visual calibration, or the next board-target move.
- Pickup-box moves need a colour-specific top waypoint. The safe pickup path is
  `waiting_pose -> pickup_top_pose -> pickup_pose -> pickup_top_pose -> waiting_pose`.
- Never move directly from `pickup_pose` to `waiting_pose`; the arm can hit the
  stone box edge.

This is a physical safety constraint: direct low-height sweeps over the board
or out of the pickup box can hit stones, the board, or the box.

## SO101 Motion Constraints

- The runtime SO101 mover is the low-level STS3215 mover in
  `src/robot/so101_lowlevel_mover.py`.
- Do not casually write servo EEPROM or motion-profile registers. Normal
  movement should only write torque enable and goal position.
- Do not replace the verified low-level motion parameters unless the user is
  explicitly debugging motion on the real arm.

## Interaction And Voice Boundary

- Keep speech, wake-word, TTS, ASR, and other human-facing interaction code
  behind the ports in `src/interaction.py`.
- Voice implementations should implement `RobotInteractionController` and be
  injected into `GameOrchestrator`; do not make the orchestrator import voice
  packages directly.
- Voice code must not depend on robot motion, perception, Rapfi, GPIO, or board
  pose-map internals. Those modules may call the interaction port, but the
  dependency must not point back into them.
- Keyboard confirmation remains the default human-turn controller unless the
  user explicitly wires a different controller.
- See `docs/adr/0001-voice-interaction-boundary.md` before changing this
  dependency direction.

## Verification

After changing live robot motion, run at least:

```bash
conda run -n lerobot python -m pytest tests/test_live_run_safety.py tests/test_robot_calibration.py tests/test_move_to_board_position.py
```

For lint-sensitive changes, also run:

```bash
conda run -n lerobot python -m ruff check scripts src tests
```
