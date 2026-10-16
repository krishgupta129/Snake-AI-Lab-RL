import asyncio
import time
from pathlib import Path
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent import Agent, train_one_step
from game import SnakeGameAI

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Snake AI Lab")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


def snapshot(agent, game, steps, started_at, elapsed_before_pause, training, scores, best_score):
    elapsed = elapsed_before_pause
    if training and started_at is not None:
        elapsed += time.monotonic() - started_at
    return {
        "type": "state",
        "snake": [[p.x, p.y] for p in game.snake],
        "food": [game.food.x, game.food.y],
        "width": game.w,
        "height": game.h,
        "block_size": 20,
        "score": game.score,
        "episode": agent.n_games,
        "steps": steps,
        "best_score": best_score,
        "scores": scores[-300:],
        "elapsed": round(elapsed, 2),
        "training": training,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Each connection owns a separate environment and model. No weights are persisted.
    agent = Agent()
    game = SnakeGameAI()
    training = True
    started_at = time.monotonic()
    elapsed_before_pause = 0.0
    steps = 0
    scores = []
    best_score = 0
    training_speed = 60  # Default to a lighter pace instead of maximum speed.
    last_send = 0.0

    async def send_state():
        await ws.send_json(snapshot(
            agent, game, steps, started_at, elapsed_before_pause,
            training, scores, best_score
        ))

    await send_state()
    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                command = message.get("command")
                if command == "speed":
                    requested = message.get("value", 60)
                    if isinstance(requested, (int, float)) and int(requested) in (20, 40, 60, 80, 100):
                        training_speed = int(requested)
                elif command == "pause" and training:
                    elapsed_before_pause += time.monotonic() - started_at
                    started_at = None
                    training = False
                elif command == "start" and not training:
                    started_at = time.monotonic()
                    training = True
                elif command == "reset":
                    agent = Agent()
                    game = SnakeGameAI()
                    training = True
                    started_at = time.monotonic()
                    elapsed_before_pause = 0.0
                    steps = 0
                    scores = []
                    best_score = 0
                elif command == "close":
                    break
            except asyncio.TimeoutError:
                pass

            if training:
                result = await asyncio.to_thread(train_one_step, agent, game)
                steps += 1
                if result["done"]:
                    finished_score = int(result["score"])
                    scores.append(finished_score)
                    best_score = max(best_score, finished_score)

            now = time.monotonic()
            if now - last_send >= 0.075:
                await send_state()
                last_send = now
            # Scale the delay inversely with the selected speed. 100% retains the
            # original 25 ms delay; lower levels take fewer steps per second.
            await asyncio.sleep(0.025 * (100 / training_speed) if training else 0.08)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        with suppress(Exception):
            await ws.close()
