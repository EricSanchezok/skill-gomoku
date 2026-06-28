# ADR-0001: Keep Voice Behind Interaction Ports

## Status
Accepted

## Context
Voice, wake-word, TTS, ASR, and other human-facing interaction work may be
developed independently from the live Gomoku loop. The core loop already owns
camera perception, Rapfi decisions, SO101 motion, air-pump control, and board
state. Mixing voice packages into those modules would make real-robot changes
and voice experiments hard to merge.

## Decision
Voice implementations must live behind the ports in `src/interaction.py`.
Production voice code should implement `RobotInteractionController` and be
injected into `GameOrchestrator` or the live entrypoint. The orchestrator,
robot, perception, AI, and GPIO modules must not import voice packages.

`KeyboardHumanTurnController` remains the default human-turn controller until a
different controller is explicitly wired in.

## Consequences

### Positive
- Voice work can happen in parallel without touching robot motion or perception.
- Tests can keep using `NullRobotInteraction` or small fake controllers.
- Hardware safety rules stay concentrated in the robot and orchestrator layers.

### Negative
- Voice implementations need a small adapter class instead of calling the game
  loop directly.

### Neutral
- The live entrypoint remains the composition root that chooses console,
  no-op, or future voice-backed interaction implementations.

## Alternatives Considered

**Import voice packages directly in `GameOrchestrator`**
- Rejected: would couple game rules and robot safety code to optional voice
  dependencies.

**Put voice commands inside `scripts/run_live_game.py`**
- Rejected: would make the script a growing integration blob and increase merge
  conflicts.

