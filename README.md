## Demo
Demo is currently hosted on Huggingface with this link: https://huggingface.co/spaces/pastelite/game-classify
## How to run this
1. Clone the repository
```
git clone https://github.com/pastelite/game_detection_ai
cd game_detection_ai
```
2. Install required requirements using
```
pip install -r requirements.txt
```
3. To run the interface of this app, you will need to download the checkpoints and place it in checkpoints folder. The checkpoints can be found here: https://huggingface.co/pastelite/game-classification
4. Start the fast API
```
uvicorn fastapi_start:app --host 0.0.0.0
```

## Build
To build this, you have to set up as mention above but this time run the ``docker build`` command. The docker file are already pre configered. You can also download the prebuilt container in docker hub here: https://hub.docker.com/r/pastelite/game_classify

## Train
To train, we recommand to use "Copy_of_train.ipynb" instead since it is currently the latest and have Google Colab stuff preconfigered.

