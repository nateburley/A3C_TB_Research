# Summer '20 UG project: When Does the Transformed Bellman Operator Help in A3C?

## About

This projects aims to study when does the Transformed Bellman Operator help in the A3C algorithm

This repo contains the implementation of A3C and A3CTB.

_Note that this code rus on CPU only and does not support GPU._

**This implementation is forked from: https://github.com/gabrieledcjr/DeepRL**

## Local Installation

To try the code in a local machine, it is recommended to use a virtual environment for installation.

Using ```conda``` as an example here.

Create a virtual environment (name "a3ctb-env") from the ```env.yml``` file:

    $ conda env create -f env.yml

Activate the environment

    $ conda activate a3ctb-en

Install other dependencies (if needed)

_Make sure your ```pip``` points to python 3.6_

    $ pip install gym atari_py coloredlogs \
    termcolor pyglet==1.5.0 tables matplotlib \
    numpy opencv-python moviepy scipy \
    scikit-image pygame pandas

Install Tensorflow 1.11. The following command installs _CPU-only_ Tensorflow; this implementation runs in CPU only.

    $ pip install tensorflow==1.11

_Note: you may use a newer version of Tensorflow (e.g., 1.14) but there will be warnings messages. However, this implementation has not been tested in Tensorflow 1.15 or higher_

## How to run in your local machine

We provide two bash files to run corresponding experiments. The bash file takes the first argument as the game input. Valid games are: _MsPacman_ and _Pong_

For example, to train MsPacman in baseline A3C,

    $./run_a3c.sh MsPacman

To train MsPacman in A3CTB,

    $./run_a3ctb.sh MsPacman

## Train on Aeolus cluster

Aeolus documentation: https://www.aeolus.wsu.edu/

Code directory: /data/datascience/UGproject

The easiest way to train on aeolus cluster is by using the _singularity_ container. See more about singularity [here](https://sylabs.io/guides/3.4/user-guide/) (aeolus supports singularity 3.4)

We provide the `production_cpu.simg` file in the directory, which contains all the dependencies needed for running the experiments (like the `env.yml` file used in a local machine). Aeolus uses `qsub` for submitting jobs, we provide two bash files for submitting corresponding experiments.

To submit jobs on A3C,

    $ qsub submit_a3c_job.sh

To submit jobs on A3CTB,

    $ qsub submit_a3ctb_job.sh

Set `game=Pong` or `game=MsPacman` inside each .sh file to switch games

Change the "email" option to YOUR email address to receive logging info about submitted jobs
