# Magic Turtle: Action Recognition for Control

This project aims at using gesture recognition for real-time control of moving "Turtle" agent. It recognizes gestures "here" (index finger) to determine in which direction agent is moving, and "thumb up" / "thumb down" to change speed of agent.

## First-time run

Pip requirements are in the `requirements.txt`. Additionally, code requires `hand_landmarker.task` file, which can be downloaded automatically during the first run of code or manually from [this link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).