#!/usr/bin/env python3
import time
import diambra.arena

fps = 20

def main():
    # Environment creation
    env = diambra.arena.make("umk3", render_mode="human")

    # Environment reset
    observation, info = env.reset(seed=42)

    # Agent-Environment interaction loop
    while True:
        time.sleep(1/fps) # To slow down the execution for better visualization
        # (Optional) Environment rendering
        env.render()

        # Action random sampling
        actions = env.action_space.sample()

        # Environment stepping
        observation, reward, terminated, truncated, info = env.step(actions)
        print("+"*10,reward)
        # Episode end (Done condition) check
        if terminated or truncated:
            observation, info = env.reset()
            break

    # Environment shutdown
    env.close()

    # Return success
    return 0

if __name__ == '__main__':
    main()