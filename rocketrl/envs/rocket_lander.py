"""RocketLander, modernized for Gymnasium with a continuous action space.

Ported from the original gym 0.21 environment. The Box2D dynamics are preserved
byte-for-byte; what changed is the control/observation interface, so the
environment is correct-by-construction for the comparative study:

  * Gymnasium API: reset() -> (obs, info); step() -> (obs, reward, terminated,
    truncated, info). Episode truncation is delegated to the TimeLimit wrapper
    via the registered max_episode_steps.
  * Continuous Box(3) action space by default (gimbal rate, throttle rate,
    side-thruster). A discrete mode is retained for ablations.
  * RAW observations. The original baked a hardcoded (state - MEAN) / VAR
    normalization into the env; that opaque constant is removed. Normalization
    is now an explicit, reproducible VecNormalize wrapper at train time.
  * Deterministic seeding through self.np_random everywhere (the original used
    the unseeded global np.random for the initial x-position and spin).
  * Landing/termination outcomes are exposed in `info` (landed, crashed,
    touchdown_speed, touchdown_angle, x_landing, fuel_used) instead of being
    printed to stdout.
"""

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape,
                      revoluteJointDef, distanceJointDef, contactListener)
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Physics constants (unchanged from the original environment)
# ---------------------------------------------------------------------------
VEL_STATE = True          # include velocities in the observation
FPS = 60
SCALE_S = 0.35            # temporal scaling, lower is faster
INITIAL_RANDOM = 0.4     # random scaling of initial velocity/spin

START_HEIGHT = 1000.0
START_SPEED = 80.0

MIN_THROTTLE = 0.4
GIMBAL_THRESHOLD = 0.4
MAIN_ENGINE_POWER = 1600 * SCALE_S
SIDE_ENGINE_POWER = 100 / FPS * SCALE_S

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 47.9
ENGINE_HEIGHT = ROCKET_WIDTH * 0.5
ENGINE_WIDTH = ENGINE_HEIGHT * 0.7
THRUSTER_HEIGHT = ROCKET_HEIGHT * 0.86

LEG_LENGTH = ROCKET_WIDTH * 2.2
BASE_ANGLE = -0.27
SPRING_ANGLE = 0.27
LEG_AWAY = ROCKET_WIDTH / 2

SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 40

VIEWPORT_H = 720
VIEWPORT_W = 504
H = 1.1 * START_HEIGHT * SCALE_S
W = float(VIEWPORT_W) / VIEWPORT_H * H

# Optional STATIC observation normalization, reproducing the original environment.
# These mean/std constants were precomputed from successful trajectories, so unlike
# a dynamic VecNormalize (which scales by stats over early high-altitude crash
# states) they scale the near-pad landing region correctly. Used only when the env
# is constructed with static_normalize=True.
STATIC_OBS_MEAN = np.array([-0.034, -0.15, -0.016, 0.0024, 0.0024, 0.137,
                            -0.02, -0.01, -0.8, 0.002], dtype=np.float32)
STATIC_OBS_STD = np.sqrt(np.array([0.08, 0.33, 0.0073, 0.0023, 0.0023, 0.8,
                                   0.085, 0.0088, 0.063, 0.076], dtype=np.float32))


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.water in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.containers[0] in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        else:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class RocketLander(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, continuous: bool = True, render_mode: str | None = None,
                 static_normalize: bool = False):
        super().__init__()
        self.continuous = continuous
        self.render_mode = render_mode
        self.static_normalize = static_normalize

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.legs = []
        self.containers = []

        # Raw observation bounds. Positional/angle/control terms are ~[-1, 1];
        # velocities are unbounded. Slightly widened to avoid boundary surprises.
        obs_high = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5,
                             np.inf, np.inf, np.inf], dtype=np.float32)
        if not VEL_STATE:
            obs_high = obs_high[:7]
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        if self.continuous:
            self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        # rendering handles (lazy)
        self._screen = None
        self._clock = None

    # -- lifecycle ----------------------------------------------------------
    def _destroy(self):
        if not self.water:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)      # seeds self.np_random deterministically
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.gimbal = 0.0
        self.landed_ticks = 0
        self.stepnumber = 0
        self.fuel_used = 0.0
        self.smoke = []

        self.terrainheigth = H / 20
        self.shipheight = self.terrainheigth + SHIP_HEIGHT
        ship_pos = W / 2
        self.helipad_x1 = ship_pos - SHIP_WIDTH / 2
        self.helipad_x2 = self.helipad_x1 + SHIP_WIDTH
        self.helipad_y = self.terrainheigth + SHIP_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((0, 0), (W, 0),
                                             (W, self.terrainheigth), (0, self.terrainheigth))),
                friction=0.1, restitution=0.0))
        self.water.color1 = rgb(70, 96, 176)

        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=(
                    (self.helipad_x1, self.terrainheigth),
                    (self.helipad_x2, self.terrainheigth),
                    (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                    (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT))),
                friction=0.5, restitution=0.0))

        self.containers = []
        for side in [-1, 1]:
            self.containers.append(self.world.CreateStaticBody(
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=(
                        (ship_pos + side * 0.95 * SHIP_WIDTH / 2, self.helipad_y),
                        (ship_pos + side * 0.95 * SHIP_WIDTH / 2, self.helipad_y + SHIP_HEIGHT),
                        (ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT,
                         self.helipad_y + SHIP_HEIGHT),
                        (ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT, self.helipad_y))),
                    friction=0.2, restitution=0.0)))
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)

        # Initial pose now uses the seeded RNG (was global np.random).
        initial_x = W / 2 + W * self.np_random.uniform(-0.3, 0.3)
        initial_y = H * 0.95
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y), angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-ROCKET_WIDTH / 2, 0),
                                             (+ROCKET_WIDTH / 2, 0),
                                             (ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                                             (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT))),
                density=1.0, friction=0.5,
                categoryBits=0x0010, maskBits=0x001, restitution=0.0))
        self.lander.color1 = rgb(230, 230, 230)

        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY, initial_y + ROCKET_WIDTH * 0.2),
                angle=(i * BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=(
                        (0, 0), (0, LEG_LENGTH / 25), (i * LEG_LENGTH, 0),
                        (i * LEG_LENGTH, -LEG_LENGTH / 20), (i * LEG_LENGTH / 3, -LEG_LENGTH / 7))),
                    density=1, restitution=0.0, friction=0.2,
                    categoryBits=0x0020, maskBits=0x001))
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(i * LEG_AWAY, ROCKET_WIDTH * 0.2), localAnchorB=(0, 0),
                enableLimit=True, maxMotorTorque=2500.0, motorSpeed=-0.05 * i, enableMotor=True)
            djd = distanceJointDef(
                bodyA=self.lander, bodyB=leg,
                anchorA=(i * LEG_AWAY, ROCKET_HEIGHT / 8),
                anchorB=leg.fixtures[0].body.transform * (i * LEG_LENGTH, 0),
                collideConnected=False, frequencyHz=0.01, dampingRatio=0.9)
            if i == 1:
                rjd.lowerAngle, rjd.upperAngle = -SPRING_ANGLE, 0
            else:
                rjd.lowerAngle, rjd.upperAngle = 0, +SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)
            self.legs.append(leg)

        self.lander.linearVelocity = (
            -self.np_random.uniform(0, INITIAL_RANDOM) * START_SPEED * 0.3 * (initial_x - W / 2) / W,
            -START_SPEED)
        self.lander.angularVelocity = (1 + INITIAL_RANDOM) * self.np_random.uniform(-1, 1)

        self.drawlist = self.legs + [self.water, self.ship] + self.containers + [self.lander]

        # Produce the first observation by applying a neutral action, as the
        # original environment did.
        neutral = np.zeros(3, dtype=np.float32) if self.continuous else 6
        obs, _, _, _, info = self.step(neutral)
        return obs, info

    # -- dynamics -----------------------------------------------------------
    def step(self, action):
        self.force_dir = 0

        if self.continuous:
            action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            self.gimbal += float(action[0]) * 0.15 / FPS
            self.throttle += float(action[1]) * 0.5 / FPS
            if action[2] > 0.5:
                self.force_dir = 1
            elif action[2] < -0.5:
                self.force_dir = -1
        else:
            if action == 0:
                self.gimbal += 0.1
            elif action == 1:
                self.gimbal -= 0.1
            elif action == 2:
                self.throttle += 0.1
            elif action == 3:
                self.throttle -= 0.1
            elif action == 4:
                self.force_dir = -1
            elif action == 5:
                self.force_dir = 1

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = 0 if self.throttle == 0.0 else MIN_THROTTLE + self.throttle * (1 - MIN_THROTTLE)

        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (-np.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power,
                 np.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power)
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        force_pos_c = self.lander.position + THRUSTER_HEIGHT * np.array(
            (np.sin(self.lander.angle), np.cos(self.lander.angle)))
        force_c = (-self.force_dir * np.cos(self.lander.angle) * SIDE_ENGINE_POWER,
                   self.force_dir * np.sin(self.lander.angle) * SIDE_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse=force_c, point=force_pos_c, wake=False)

        self.world.Step(1.0 / FPS, 60, 60)

        pos = self.lander.position
        vel_l = np.array(self.lander.linearVelocity) / START_SPEED
        vel_a = self.lander.angularVelocity
        x_distance = (pos.x - W / 2) / W
        y_distance = (pos.y - self.shipheight) / (H - self.shipheight)

        angle = (self.lander.angle / np.pi) % 2
        if angle > 1:
            angle -= 2

        state = [
            2 * x_distance,
            2 * (y_distance - 0.5),
            angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            2 * (self.throttle - 0.5),
            (self.gimbal / GIMBAL_THRESHOLD),
        ]
        if VEL_STATE:
            state.extend([vel_l[0], vel_l[1], vel_a])

        # --- reward (unchanged shaping) ------------------------------------
        distance = np.linalg.norm((3 * x_distance, 3 * y_distance))
        speed = np.linalg.norm(vel_l)
        groundcontact = self.legs[0].ground_contact or self.legs[1].ground_contact
        brokenleg = (self.legs[0].joint.angle < 0 or self.legs[1].joint.angle > -0) and groundcontact
        outside = abs(pos.x - W / 2) > W / 2 or pos.y > H
        fuelcost = 0.1 * (0 * self.power + abs(self.force_dir)) / FPS
        self.fuel_used += fuelcost
        landed_now = self.legs[0].ground_contact and self.legs[1].ground_contact and speed < 0.1
        terminated = False
        landed = False

        reward = -fuelcost
        if outside or brokenleg:
            self.game_over = True

        if self.game_over:
            terminated = True
        else:
            shaping = -0.5 * (distance + speed + abs(angle) ** 2)
            shaping += 0.1 * (self.legs[0].ground_contact + self.legs[1].ground_contact)
            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping
            self.prev_shaping = shaping

            if landed_now:
                self.landed_ticks += 15
            else:
                self.landed_ticks = 0
            if self.landed_ticks >= FPS:
                reward = 1.0
                terminated = True
                landed = True

        if terminated:
            reward += max(-1, 0 - 2 * (speed + distance + abs(angle) + abs(vel_a)))
        elif not groundcontact:
            reward -= 0.25 / FPS

        reward = float(np.clip(reward, -1, 1))
        self.stepnumber += 1

        info = {}
        if terminated:
            crashed = self.game_over and not landed
            info = {
                "landed": landed,
                "crashed": crashed,
                "touchdown_speed": float(speed),
                "touchdown_angle": float(abs(angle)),
                "x_landing": float(x_distance),
                "fuel_used": float(self.fuel_used),
            }

        obs = np.array(state, dtype=np.float32)
        if self.static_normalize:
            obs = ((obs - STATIC_OBS_MEAN[:len(obs)]) / STATIC_OBS_STD[:len(obs)]).astype(np.float32)
        if self.render_mode == "human":
            self.render()
        # Truncation is applied by the TimeLimit wrapper, not here.
        return obs, reward, terminated, False, info

    # -- rendering ----------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        scale = VIEWPORT_H / H
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            else:
                self._screen = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
            self._clock = pygame.time.Clock()

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        surf.fill((126, 150, 233))

        def to_screen(v):
            return int(v[0] * scale), int(VIEWPORT_H - v[1] * scale)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [to_screen(trans * v) for v in f.shape.vertices]
                color = tuple(int(255 * c) for c in getattr(obj, "color1", (0.9, 0.9, 0.9)))
                pygame.draw.polygon(surf, color, path)
                pygame.draw.polygon(surf, (0, 0, 0), path, 1)

        # Engine nozzle and exhaust flame. These are visual-only primitives (not
        # Box2D bodies): the flame is anchored at the rocket base, rotated by the
        # gimbal relative to the body, then carried through the body's own
        # rotation/translation. Flame length scales with the main-engine power.
        def engine_frame_to_screen(local_points):
            gimbal_cos = np.cos(self.gimbal)
            gimbal_sin = np.sin(self.gimbal)
            body_angle = self.lander.angle
            body_cos = np.cos(body_angle)
            body_sin = np.sin(body_angle)
            base_x = self.lander.position[0]
            base_y = self.lander.position[1]
            screen_points = []
            for local_x, local_y in local_points:
                gimbal_x = gimbal_cos * local_x - gimbal_sin * local_y
                gimbal_y = gimbal_sin * local_x + gimbal_cos * local_y
                world_x = body_cos * gimbal_x - body_sin * gimbal_y + base_x
                world_y = body_sin * gimbal_x + body_cos * gimbal_y + base_y
                screen_points.append(to_screen((world_x, world_y)))
            return screen_points

        if self.power > 0:
            # Enlarged beyond the physical nozzle size so the exhaust reads
            # clearly at the zoomed-out whole-descent viewport scale.
            flame_width = ENGINE_WIDTH * 2.5
            flame_length = ENGINE_HEIGHT * (1 + 18 * self.power)
            flame_points = [
                (flame_width * 0.5, -ENGINE_HEIGHT),
                (-flame_width * 0.5, -ENGINE_HEIGHT),
                (-flame_width, -flame_length * 0.6),
                (0, -flame_length),
                (flame_width, -flame_length * 0.6),
            ]
            # bright outer flame with a hotter inner core
            pygame.draw.polygon(surf, (255, 170, 40), engine_frame_to_screen(flame_points))
            inner_points = [
                (flame_width * 0.28, -ENGINE_HEIGHT),
                (-flame_width * 0.28, -ENGINE_HEIGHT),
                (0, -flame_length * 0.7),
            ]
            pygame.draw.polygon(surf, (255, 240, 170), engine_frame_to_screen(inner_points))

        engine_nozzle_points = [
            (0, 0),
            (ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
            (-ENGINE_WIDTH / 2, -ENGINE_HEIGHT),
        ]
        pygame.draw.polygon(surf, (100, 100, 100), engine_frame_to_screen(engine_nozzle_points))

        # Side-thruster puff, drawn opposite to the applied lateral impulse.
        if self.force_dir != 0:
            thruster_points = [
                (self.force_dir * ROCKET_WIDTH * 0.5, THRUSTER_HEIGHT),
                (self.force_dir * ROCKET_WIDTH * 2.0, THRUSTER_HEIGHT * 1.03),
                (self.force_dir * ROCKET_WIDTH * 2.0, THRUSTER_HEIGHT * 0.97),
            ]
            pygame.draw.polygon(surf, (200, 200, 200), engine_frame_to_screen(thruster_points))

        self._screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self._clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            return None
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
