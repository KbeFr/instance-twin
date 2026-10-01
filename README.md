# Instance Twin

The lower level of the hierarchy: one container, one twin, at most one agent. An instance twin
starts **unbound**. Once the aggregate pairs it with an agent, it builds a live model of that agent
from the discovery spec, estimates the agent's state, runs its local controller and reports to the
aggregate.

## Lifecycle

```
UNBOUND ──── pairing handshake ────► BOUND ──── CANCEL / agent or aggregate lost ────► UNBOUND
discovery + heartbeat                10 Hz step · twin_state doubles as heartbeat
```

When bound, the twin:
1. builds an `AgentProfile`
2. subscribes to the agent's channels
3. sends an `InitialCommandMessage` on the agent's `action` topic, retrying every ~2 s until the first sensor message confirms the link
4. starts stepping

When released, it stops the agent, clears its state, and becomes a candidate again.

## Agent model (`AgentProfile`)

Every part is built from the resolved discovery through name-based registries. A new model plugs in with one decorator.

| Part | Registered names | Location |
|---|---|---|
| Kinematics | `diff`, `omni`, `acker` | `irsim_borrowed/kinematics` |
| Geometry | `circle`, `rectangle`, `polygon`, `linestring`, `map` | `irsim_borrowed/geometry` |
| Controller | `c3bf`, `cbf`, `apf`, `PPC` | `irsim_borrowed/controller` |
| Estimator | `ekf` | `estimation/` |
| State sensors | `pose`, `gps`, `odom`, `wheel_odom`, `imu` | `sensors/state_sensors.py` |
| Perception sensors | `aruco`, `sim2d_object`, `lidar` | `sensors/perception_sensors.py` |
| Battery | `two_friction`, `anc_only`, `dummy` | `battery/` |
| Agent interface | `simulated`, `ros2`, `turtlebot4`, `turtlebot4_jazzy` | `interfaces/` |

- **State sensors** turn a message into named components (`x, y, theta, vx, vy, wz`), each with a noise std. Only components with a configured noise are fused. At least one sensor must provide an absolute position, or the estimate never anchors.
- **EKF:** predicts with the agent's kinematics and a first-order velocity lag. Jacobians are numeric, so any kinematics and sensor combination works. Chi-square gating is optional.
- **Perception sensors** transform detections from the sensor frame (body pose ∘ mount offset) into world-frame `ObstacleObservation`s. They only do so once the estimate is anchored.
- **Agent interface** handles the wire format:
  - `simulated` passes `core_msgs` through unchanged.
  - `ros2` and `turtlebot4` decode ROS 2 (CDR) messages from the flexIA bridge (Odometry, Imu, BatteryState, LaserScan, ArucoMarkers) and encode Twist or TwistStamped.
  - Binding is refused if the interface cannot carry a channel the sensors need.

## Step loop (10 Hz)

1. Drain the inbox (mission envelopes and agent sensor data).
2. Drop stale obstacles.
3. Run the EKF prediction up to now.
4. The controller picks the next command toward the current mission goal. The command stays zero until the estimate is anchored.
5. Step the battery model.
6. Check for arrival and mission completion.
7. Publish `twin_state` and any new `obstacle`s.
8. If `TWIN_DRIVE_AGENT` is on, send the command to the agent.

The twin bids time and battery cost (from its battery model) on missions and follows the aggregate's A* path point by point.

## Required discovery fields

Binding is refused unless the resolved spec provides all of:
- `agent_id`
- `kind`
- `interface_name`
- `radius`
- `max_speed`
- `shape`
- `kinematics`
- `state_sensors`

```yaml
agent_type: simulated                     # selects the agent interface
kinematics: {name: diff}
shape: {name: circle, radius: 0.2}
radius: 0.2
max_speed: 1.5
mass: 1.0
friction: 0.25
avg_speed: 1.0                            # mass/friction/avg_speed are required with a battery
state_sensors:
  - {name: pose, noise: {x: 0.02, y: 0.02, theta: 0.02}}
  - {name: wheel_odom, track_width: 0.2, noise: {vx: 0.05, wz: 0.1}}
perception_sensors:
  - name: aruco
    offset: {position: {x: 0.0, y: 0.0, z: 0.3}, orientation: {x: 0, y: 0, z: 0, w: 1}}
battery: {name: two_friction, init: 100.0, anc_drain: 5.0, scale: 1, capacity: 71928.0}
controller: {name: c3bf, robot_type: diff, safety_margin: 0.05, goal_gain: 0.8}
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `TWIN_NAME` | `$HOSTNAME` | Instance name (unique) |
| `TWIN_NAMESPACE` | `default-ns` | Topic namespace |
| `TWIN_TICK_HZ` | `10` | Step rate |
| `TWIN_DRIVE_AGENT` | `0` | Send motion commands to the agent. **Off by default** |
| `TWIN_GUI` / `TWIN_GUI_PORT` | `1` / `8081` | Read-only viewer |
| `MQTT_BROKER_HOST` / `REDIS_HOST` | `localhost` | Transport |

Config files: `config/config.yaml` (flexNode and heartbeat), `global_topic_config.yaml`, `aggregate_topic_config.yaml`.

## Run

```bash
pip install -e core_msgs -e flexCommunicator -e instance-twin
TWIN_NAME=InstanceTwin1 instance-twin
```

`irsim_borrowed/` contains adapted code from [IR-SIM](https://github.com/hanruihua/ir-sim) (R. Han); see its README.