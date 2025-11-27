# Mortal Kombat 3 Reinforcement Learning

## Setup

#### 1. Set up Python Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

#### 2. Install DIAMBRA Dependencies

```bash
uv pip install diambra
uv pip install diambra-arena
```

#### 3. Configure ROM Directory

Create the DIAMBRA roms directory:

```bash
mkdir -p /home/[USERNAME]/.diambra/roms/
```
> You can use any other directory on your system as well, but then you need to set a system environment variable `DIAMBRAROMSPATH=/absolute/path/to/roms/folder/`

tip : For ROMs , find the game on diambra docs and check for search keywords and feed them to Gr0k , verify SHA  ~ worked for mortal kombat 3  :)

Copy your UMK3 ROM file to the directory:

```bash
cp ~/path/to/umk3.zip /home/[USERNAME]/.diambra/roms/
```

#### 4. Verify ROM Installation

Check if the ROM is properly installed:

```bash
diambra arena check-roms umk3.zip
```
> If you are using a custom directory, and have set the environment varible correctly, this should run without errors as well. Try restarting the terminal if it doesn't.

✅ This command must pass successfully before proceeding. -_-

#### 5. Register DIAMBRA Account

On first run, you'll need to register for a DIAMBRA account:

1. Run the test script: `diambra run python trial.py` (You need docker running)
2. When prompted to register, visit [diambra.ai](https://diambra.ai) (cli gives diambra.ai/register it wont work)
3. Click "Login" (top right) and create an account
4. Complete the registration process

## 🎮 Usage

See the specific branches for [PPO](https://github.com/Reckadon/ReinforcementLearning-MortalKombat-3/tree/ppo), [DQN](https://github.com/Reckadon/ReinforcementLearning-MortalKombat-3/tree/dqm) and [Actor-Critic](https://github.com/Reckadon/ReinforcementLearning-MortalKombat-3/tree/a2c) respectively for their usages and documentation.


## 🔗 Resources

- [DIAMBRA Arena Documentation](https://diambra.ai/docs/)

---
