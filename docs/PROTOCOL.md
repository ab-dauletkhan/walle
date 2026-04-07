# WALL-E Firmware Serial Protocol

Communication between the Python host (Jetson/Raspberry Pi) and the Arduino firmware.

## Physical Layer

- **Interface:** USB Serial (UART)
- **Baud rate:** 115200 (configurable via `WALLE_BAUD_RATE`)
- **Line ending:** `\n`

## Command Format

```
<letter>[<number>]\n
```

- Single uppercase letter = axis/servo identifier
- Optional signed integer (up to 3 digits): `-100` to `100`
- Arduino responds with `OK\n` within 2 seconds (ACK timeout)

### Command Table

| Prefix | Target | Range | Example | Description |
|--------|--------|-------|---------|-------------|
| `G` | Head rotation | 0–100 | `G50` | Pan head left (0) → center (50) → right (100) |
| `T` | Neck top servo | 0–100 | `T80` | Upper neck joint position |
| `B` | Neck bottom servo | 0–100 | `B20` | Lower neck joint position |
| `L` | Left arm | 0–100 | `L50` | Left arm position (0=down, 100=up) |
| `R` | Right arm | 0–100 | `R50` | Right arm position (0=down, 100=up) |
| `Y` | Drive (fwd/back) | -100–100 | `Y50`, `Y-50` | Positive=forward, negative=backward |
| `X` | Turn (left/right) | -100–100 | `X-50`, `X50` | Negative=left, positive=right |
| `q` | Emergency stop | — | `q` | Stop all motor movement immediately |
| `A0` | Animation | 0 | `A0` | Play neutral/reset animation |
| `?` | Status query | — | `?` | Request current servo positions |

### Multi-Command

Commands can be sent in a single write separated by `\n`:
```
L80\nR80\n
```
Each sub-command gets its own ACK.

## Status Response

Query: `?\n`

Response format:
```
STATUS <s1>,<s2>,<s3>,<s4>,<s5>,<s6>,<s7> M<mx>,<my>
```

Where `s1`–`s7` are raw servo microsecond values and `mx`,`my` are motor speeds.

## Tool Actions (Stable API)

These are the Python-level tool names exposed to the LLM. Each maps to one or more serial commands.

| Tool Name | Commands Sent | Description |
|-----------|---------------|-------------|
| `set_head_rotation` | `G{0-100}` | Pan head |
| `set_neck_position` | `T{pos}` + `B{100-pos}` | Coordinated dual-neck servo |
| `set_both_arms` | `L{val}\nR{val}` | Set both arms |
| `drive_forward` | `Y{speed}` [+ `q` after duration] | Drive forward |
| `drive_backward` | `Y{-speed}` [+ `q` after duration] | Drive backward |
| `turn_left` | `X{-speed}` [+ `q` after duration] | Turn left |
| `turn_right` | `X{speed}` [+ `q` after duration] | Turn right |
| `stop_movement` | `q` | Emergency stop |
| `express_emotion` | Pose: T+B+L+R | Set expressive pose (happy/sad/neutral) |
| `scan_surroundings` | `G20` → `G50` → `G80` → `G50` | Head sweep |
| `wave_hello` | `R80` → `R35` → `R80` (×2) | Wave with right arm |
| `reset_to_neutral` | `A0` | Reset all servos via animation |

## Poses

| Emotion | Neck Top | Neck Bottom | Left Arm | Right Arm |
|---------|----------|-------------|----------|-----------|
| Happy | 80 | 20 | 80 | 80 |
| Sad | 20 | 80 | 20 | 20 |
| Neutral | 50 | 50 | 40 | 40 |

## Validation

The `SerialManager` validates commands before sending:
- Must start with a letter (`[A-Za-z]`)
- Optional sign + up to 3 digits
- Maximum 5 characters total
- Invalid commands return `Error: invalid command '...'` without hitting the serial port
