#!/usr/bin/env python3
import time
import diambra.arena
from diambra.arena import SpaceTypes, Roles, EnvironmentSettings

def make_env():
    settings = EnvironmentSettings()
    settings.step_ratio = 4
    settings.frame_shape = (128, 128, 0)   # resized RGB frames
    # settings.frame_shape = (0, 0, 1)    # grayscale original size (example)
    # settings.frame_shape = (0, 0, 0)    # deactivated (original RGB)
    settings.action_space = SpaceTypes.DISCRETE     # or SpaceTypes.MULTI_DISCRETE
    settings.role = Roles.P1
    settings.continue_game = 0.0
    settings.show_final = False
    settings.difficulty = 2      # 1..5 for UMK3 - pick what you want
    settings.characters = "Kitana"
    env = diambra.arena.make("umk3", settings, render_mode="human")
    return env

def run_episode(env):
    obs, info = env.reset(seed=int(time.time() % 2**31))
    env.show_obs(obs)
    total_reward = 0.0
    steps = 0

    while True:
        action = env.action_space.sample()   # replace with your policy output
        # If using MULTI_DISCRETE you may get a list/tuple here; ensure your policy matches.
        # print("Action (sampled):", action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        env.show_obs(obs)   # shows current frame (if render_mode supports it)

        done = terminated or truncated
        if done:
            return total_reward, steps, info

def main():
    env = make_env()
    try:
        # single episode example; wrap for training loop
        total_reward, steps, info = run_episode(env)
        print(f"Episode finished — reward={total_reward:.2f}, steps={steps}")
        # Example of resetting with explicit option changes:
        options = {
            "role": Roles.P1,        # keep agent as P1 (or Roles.P2 / None)
            "difficulty": 4,         # set a stronger CPU if you want
            # don't set keys to None unless docs say so
            # "characters": "Kitana",
            # "outfits": 4,          # include only if supported for the game
        }
        obs, info = env.reset(options=options)
        env.show_obs(obs)
        # optionally run more episodes or hand `env` to your trainer/loop
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()
