# Snake AI Lab

A Streamlit demo of a Snake agent learning with Deep Q-Learning (PyTorch).

## Features

- Live Snake rendering inside Streamlit, without a Pygame desktop window
- Elapsed training timer
- Episode score and rolling-average graph
- Start, pause, and reset controls
- Model weights are kept in memory and are not saved to disk

## Windows setup (no Conda)

Use Python 3.10 or 3.11 for a straightforward local setup.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- Training is performed in the active app session. Resetting the session or restarting the app discards the in-memory model.
- This demo deliberately does not call `torch.save()` or write a trained model file.
- Streamlit reruns are used to advance training in small steps; the graph adds a point at the end of each episode.
- This is a demonstration project, not a claim that the agent reaches a particular score or performance level.

## Deploy to Streamlit Community Cloud

1. Push these project files to a GitHub repository.
2. Open Streamlit Community Cloud and connect the repository.
3. Set the main file path to `streamlit_app.py`.
4. Deploy. The platform installs packages from `requirements.txt`.

For a public deployment, begin with conservative training refresh settings. Cloud CPU and memory limits may affect training speed.
