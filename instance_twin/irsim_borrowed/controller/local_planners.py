import numpy as np

from instance_twin.irsim_borrowed.controller.controller_handler import register_controller, ControllerModel


@register_controller("apf")
class APFController(ControllerModel):
    """
    Artificial Potential Field (APF) Local Planner.
    The goal attracts the robot, while obstacles repel it.

    Great for demonstrating local minima in U-shaped obstacles!
    (Assumes Omnidirectional Kinematics)
    """

    def __init__(self, k_att=1.5, k_rep=2.0, influence_dist=2.5, max_speed=1.0):
        super().__init__()
        self.k_att = k_att  # Attractive force gain
        self.k_rep = k_rep  # Repulsive force gain
        self.influence_dist = influence_dist  # Distance at which obstacles start repelling
        self.max_speed = max_speed  # Speed limit

    def get_action(self, robot, obstacles):
        if robot.goal is None:
            return np.zeros(2)

        pos = robot.state[:2, 0]
        goal = robot.goal[:2, 0]

        # 1. Attractive Force (Pull towards goal)
        f_att = self.k_att * (goal - pos)

        # 2. Repulsive Force (Push away from obstacles)
        f_rep = np.zeros(2)
        for obs in obstacles:
            # Ignore unobstructed or invalid objects
            if getattr(obs, 'unobstructed', False):
                continue

            obs_pos = obs.position[:, 0]
            dist = np.linalg.norm(pos - obs_pos)

            # Approximate the safe distance using the radii of both objects
            # (Works as a safe fallback even for rectangular objects)
            obs_radius = getattr(obs, 'radius', 1.0)
            safe_dist = dist - robot.radius - obs_radius

            # If inside the influence bubble, calculate repulsion
            if 0.01 < safe_dist < self.influence_dist:
                # Magnitude of repulsion grows exponentially as you get closer
                rep_mag = self.k_rep * (1.0 / safe_dist - 1.0 / self.influence_dist) * (1.0 / safe_dist ** 2)
                # Direction is directly away from the obstacle
                rep_dir = (pos - obs_pos) / dist
                f_rep += rep_mag * rep_dir

        # 3. Total Force (Desired Velocity)
        total_force = f_att + f_rep

        # 4. Limit Speed
        speed = np.linalg.norm(total_force)
        if speed > self.max_speed:
            total_force = (total_force / speed) * self.max_speed

        return total_force


@register_controller("PPC")
class PurePursuitController(ControllerModel):
    """
    Simple Waypoint Follower.
    Blindly drives toward the target waypoint, ignoring obstacles entirely.

    Use this to prove that your Global Planner's path is flawless.
    (Assumes Omnidirectional Kinematics)
    """

    def __init__(self, k_p=1.5, max_speed=1.0):
        super().__init__()
        self.k_p = k_p
        self.max_speed = max_speed

    def get_action(self, robot, obstacles):
        if robot.goal is None:
            return np.zeros(2)

        pos = robot.state[:2, 0]
        goal = robot.goal[:2, 0]

        # Calculate vector to the goal
        direction = goal - pos
        dist = np.linalg.norm(direction)

        if dist < 0.05:
            return np.zeros(2)

        # Proportional control
        vel = self.k_p * direction

        # Cap to max speed
        speed = np.linalg.norm(vel)
        if speed > self.max_speed:
            vel = (vel / speed) * self.max_speed

        return vel