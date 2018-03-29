# TensorFlow starting point

This repo contains a samle Lyrics generator.

# Installation
Requirements
- You need to have [Docker](https://docs.docker.com/engine/installation/) installed

# Run

Run in root folder,
~~~~
docker-compose build && docker-compose up -d
~~~~

Login to the container,
~~~~
docker exec -it ai /bin/bash -c "TERM=$TERM exec bash"
~~~~

Run
~~~~
python main.py
~~~~

# Some things to consider

This is used as a starting point for machine learning projects. For this reason,
Keras and some other libraries come pre-installed. You may remove them at your
own discretion.

The model is trained with a handful of lyrics to avoid copyright infringement stuff. You may improve the accuracy by adding more lyrics files.

# Maintainer
[Thanos Nokas](https://www.linkedin.com/in/thanosnokas)