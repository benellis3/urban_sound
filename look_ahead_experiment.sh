#! /bin/bash
nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=0 -v $(pwd)/outputs:/source/urban_sound/outputs/ -v $(pwd)/data/UrbanSound8K:/source/urban_sound/data/UrbanSound8K urban_sound:run python src/urban_sound/main.py +dataset=urban_sound ++look_ahead=12 ++tag=look_ahead_12 &

nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=1 -v $(pwd)/outputs:/source/urban_sound/outputs/ -v $(pwd)/data/UrbanSound8K:/source/urban_sound/data/UrbanSound8K urban_sound:run python src/urban_sound/main.py +dataset=urban_sound ++look_ahead=60 ++tag=look_ahead_60 &

nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=2 -v $(pwd)/outputs:/source/urban_sound/outputs/ -v $(pwd)/data/UrbanSound8K:/source/urban_sound/data/UrbanSound8K urban_sound:run python src/urban_sound/main.py +dataset=urban_sound ++look_ahead=300 ++tag=look_ahead_300 &

nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=3 -v $(pwd)/outputs:/source/urban_sound/outputs/ -v $(pwd)/data/UrbanSound8K:/source/urban_sound/data/UrbanSound8K urban_sound:run python src/urban_sound/main.py +dataset=urban_sound ++look_ahead=3 ++tag=look_ahead_3 &

nvidia-docker run -e NVIDIA_VISIBLE_DEVICES=4 -v $(pwd)/outputs:/source/urban_sound/outputs/ -v $(pwd)/data/UrbanSound8K:/source/urban_sound/data/UrbanSound8K urban_sound:run python src/urban_sound/main.py +dataset=urban_sound ++look_ahead=1200 ++tag=look_ahead_1200 &


