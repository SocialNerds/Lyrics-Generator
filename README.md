# Lyrics Generator with Keras

This repo contains a simle Lyrics generator. We create an LSTM model with Keras and train it with some existing lyrics to produce a new song. It is the source code for this tutorial (GR)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/AvCwJ9c-JgI/0.jpg)](https://www.youtube.com/watch?v=AvCwJ9c-JgI)

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

The model is trained with a handful of lyrics to avoid copyright infringement stuff. You may improve the accuracy by adding more lyrics.

# Maintainer
[Thanos Nokas](https://www.linkedin.com/in/thanosnokas)