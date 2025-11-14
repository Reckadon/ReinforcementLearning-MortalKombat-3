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

Copy your UMK3 ROM file to the directory:

```bash
cp ~/path/to/umk3.zip /home/[USERNAME]/.diambra/roms/
```

#### 4. Verify ROM Installation

Check if the ROM is properly installed:

```bash
diambra arena check-roms umk3.zip
```

✅ This command must pass successfully before proceeding. -_-

#### 5. Register DIAMBRA Account

On first run, you'll need to register for a DIAMBRA account:

1. Run the test script: `diambra run -r /home/[USERNAME]/.diambra/roms/ python trial.py`
2. When prompted to register, visit [diambra.ai](https://diambra.ai) (cli gives diambra.ai/register it wont work)
3. Click "Login" (top right) and create an account
4. Complete the registration process

## 🎮 Usage

Run the RL training script:

```bash
diambra run -r /home/[USERNAME]/.diambra/roms/ python trial.py
```

## 🔗 Resources

- [DIAMBRA Arena Documentation](https://diambra.ai/docs/)

---