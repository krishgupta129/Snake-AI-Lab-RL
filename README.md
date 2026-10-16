# Snake AI Lab (FastAPI + Canvas)

A browser-based Snake DQN demo. Python/PyTorch performs the learning; HTML Canvas animates the game smoothly in the browser.

## What changed

- Reuses the project's `Agent`, `Linear_QNet`, `QTrainer`, and headless `SnakeGameAI`.
- FastAPI WebSocket sends compact game-state updates instead of sending a newly rendered image on every update.
- Browser renders at the display's refresh rate and interpolates between server snapshots.
- Every browser WebSocket connection creates its own model, game, replay memory, timer, and score history.
- No model weights are written to disk. Closing the tab disconnects and discards that session.
- Start, pause, and reset controls; timer, episode/best/current score, and episode-score/rolling-average graph.

## Run on Windows (no Conda)

Use Python 3.10 or 3.11. Open Command Prompt or PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

If `py` is unavailable, use `python -m venv .venv` instead. PyTorch installation can take a while because it is a large dependency.

## Deploy publicly

### Render (Python backend + frontend served together)

1. Push the contents of this folder to a GitHub repository.
2. Create a new **Web Service** on Render and connect that repository.
3. Use these settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Deploy and open the generated service URL.

The same service serves the website and WebSocket, so there is no separate frontend URL to configure. WebSockets require a running web service. Free instances may sleep when idle, take time to start, and have limited CPU/RAM. Concurrent visitors each create a separate PyTorch model and replay buffer, so begin with a small number of simultaneous users. If deployment runs out of memory or is slow, use a suitable paid instance or limit concurrent sessions.

## Project structure

- `server.py`: FastAPI app and per-connection training sessions
- `static/index.html`: responsive UI, Canvas animation, timer, controls, and chart
- `agent.py`: DQN agent and one-step training routine
- `model.py`: PyTorch network and trainer
- `game.py`: headless Snake environment

## Important behavior

- The model is randomly initialized for every new connection and starts learning from scratch.
- Pause stops training and pauses the timer. Reset creates a fresh agent and environment.
- Disconnecting or restarting the service discards training progress.
- This is a demonstration project. Learning quality and speed depend on CPU capacity, random exploration, and session duration.


### Training speed control

The browser includes five training pace settings: 20%, 40%, 60%, 80%, and 100%. New sessions default to 60% to avoid immediately running at maximum pace. The setting controls the delay between Python training steps; 100% retains the original maximum pace, while lower settings reduce the step rate and CPU pressure. Model weights remain in memory only.
