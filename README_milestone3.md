# IFT6758 Repo Template

This template provides you with a skeleton of a Python package that can be installed into your local machine.
This allows you access your code from anywhere on your system if you've activated the environment the package was installed to.
You are encouraged to leverage this package as a skeleton and add all of your reusable code, functions, etc. into relevant modules.
This makes collaboration much easier as the package could be seen as a "single source of truth" to pull data, create visualizations, etc. rather than relying on a jumble of notebooks.
You can still run into trouble if branches are not frequently merged as work progresses, so try to not let your branches diverge too much!

Also included in this repo is an image of the NHL ice rink that you can use in your plots.
It has the correct location of lines, faceoff dots, and length/width ratio as the real NHL rink.
Note that the rink is 200 feet long and 85 feet wide, with the goal line 11 feet from the nearest edge of the rink, and the blue line 75 feet from the nearest edge of the rink.

<p align="center">
<img src="./figures/nhl_rink.png" alt="NHL Rink is 200ft x 85ft." width="400"/>
<p>

The image can be found in [`./figures/nhl_rink.png`](./figures/nhl_rink.png).

## Installation

**Note: this is slightly different than the original rep you were working in!**

To install this package, first setup your Python environment (next section), and then simply run

    pip install -e ift6758

assuming you are running this from the root directory of this repo. 

## Docker

Follow the instructions [here](https://docs.docker.com/get-docker/) to install Docker on your system.
Included in this repo is a Dockerfile with brief explanations of the commands you will need to use
to use to define your Docker image.
One thing to note is that its generally a better idea to stick to using `pip` instead of Conda
environments in Docker containers, as this approach is generally more lightweight than Conda.
A common pattern is to copy the `requirements.txt` file into the docker container and then simply 
do `pip install -r requirements.txt` (done in the Dockerfile).
You can also copy an entire Python package (e.g. the `ift6758/` folder) into the container, and also
do a simple `pip install -e ift6758/` (also from the Dockerfile).

In addition, below are a list of useful docker commands that you will likely need.

### docker build ([ref](https://docs.docker.com/engine/reference/commandline/build/))
Builds a docker image directly the local Dockerfile (in the same directory):

```bash
# docker build -t <TAG>:<VERSION> .
# eg: 
docker build -t ift6758/serving:1.0.0 .
```

### docker images ([ref](https://docs.docker.com/engine/reference/commandline/images/))

Lists all images that are built.


### docker run ([ref](https://docs.docker.com/engine/reference/commandline/run/))
To run the docker image you just created, you could run:

```bash
#  docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
docker run -it --expose 127.0.0.1:8890:8890/tcp --env DOCKER_ENV_VAR=$LOCAL_ENV_VAR ift6758/serving:0.0.1 
```

In this example, `-it --expose 127.0.0.1:8890:8890/tcp --env DOCKER_ENV_VAR=$LOCAL_ENV_VAR` are the
`[OPTIONS]`, `ift6758/serving:0.0.1` is the `IMAGE`, and there are no `[COMMAND]` or `[ARG...]`.
If you run this command the docker container will run whatever you specified at the `CMD` in your 
Dockerfile; if this is not specified or it crashes, your container will immediately stop.
You could alternatively specify a different command; for example setting `[COMMAND]` to `bash` will
drop you into a bash shell in the container. 
From there you can poke around to potentially debug your app.

Some useful run options are:

- `-it`: Allocate a pseudo TTY connected to the container's STDIN (i.e. interactive mode)
- `-p/--expose`: Expose port (e.g. `-p 127.0.0.1:80:8080/tcp` binds port `8080` of the container TCP port `80` on `127.0.0.1` of the host machine.
- `-e/--env`: Set environment variable (e.g. `-e DOCKER_ENV_VAR=${LOCAL_ENV_VAR}`)
- `-d/--detach`: runs container in the background

The documentation for all of these can be found in the [official Docker docs](https://docs.docker.com/engine/reference/commandline/run)

### docker ps ([ref](https://docs.docker.com/engine/reference/commandline/ps/))

Lists running containers.
For example, if you did a `docker run` command in detached mode (with the `-d` flag), your container
will be running in the background.
To verify that your container is running, you could run `docker ps`: 

```bash
> docker ps
CONTAINER ID   IMAGE                   COMMAND   CREATED         STATUS         PORTS                      NAMES
0237661ace81   ift6758/serving:0.0.1   "bash"    4 seconds ago   Up 3 seconds   127.0.0.1:8890->8890/tcp   sleepy_jang
```

### docker exec ([ref](https://docs.docker.com/engine/reference/commandline/exec/))

This runs a command in the container.
You could use this to drop into a shell in a detached but running container, eg:

```bash
> docker exec -it sleepy_jang bash
root@0237661ace81:/code# 
```

### docker network ([ref](https://docs.docker.com/engine/reference/commandline/network/))

Allows you to ping your docker network. You can do:

```bash
> docker network ls
NETWORK ID     NAME                 DRIVER    SCOPE
15742e644eb4   bridge               bridge    local
60c57e381d21   host                 host      local
```

to see all of your existing networks.
If you didn't build with docker compose, chances are your running containers are living on the
`bridge` network.
You can then do:

```bash
> docker network inspect bridge
...
"Containers": {                                                                                                                                                                                             
            "<...some id...>": {                                                                                                                                   
                "Name": "sleepy_jang",                                                                                                                                                                          
                "EndpointID": "...",
                "MacAddress": "...",
                "IPv4Address": "172.17.0.1",  #  <--- this is the ip of the container on the docker network!
                "IPv6Address": ""
            }
...
```

or any other network you may want to inspect to get more information about the containers attached.
For example, you can find the IP of a container by the NAMES.
This is the IP of the container on the **docker network**; i.e. if you were trying to make an
HTTP request from *within* the docker network, rather than from your local host.
This may be useful to you for debugging the final part of Milestone 3, where you will put
your jupyter notebook into a container and then query the prediction service that lives in another
docker container.

**Note when using docker compose**

Docker compose does some nice name resolution stuff for you by default.
You'll notice the format of a `docker-compose.yaml` file is along the lines of:

```yaml
services:
    service1:
        ...
    service2:
        ...
```

Say your jupyter notebook lives in `service2`, and you want to make an HTTP request to `service1`.
You actually don't need to look for the container IP of `service1` - you can simply make an 
HTTP request to `http://service1:PORT/endpoint`.
The name resolution is taken care for you if you're using docker compose (but you do need to keep
track of the port).

### docker-compose ([ref](https://docs.docker.com/compose/))

_The following is taken directly from the docker compose reference_

Compose is a tool for defining and running multi-container Docker applications. 
With Compose, you use a YAML file to configure your application’s services. 
Then, with a single command, you create and start all the services from your configuration.

Using Compose is basically a three-step process:

- Define your app’s environment with a Dockerfile so it can be reproduced anywhere.
- Define the services that make up your app in docker-compose.yml so they can be run together in an isolated environment.
- Run `docker-compose up` and the Docker compose command starts and runs your entire app.

Install it with:

```bash
pip install docker-compose
```

You can then simply do `docker-compose up` to build and run the application that you sepcified in
the `docker-compose.yaml` file.


## Flask

To run the app, if you are in the same directory as `app.py`, you can run the app from the command 
line using gunicorn (Unix) or waitress (Unix + Windows):
    
```bash
gunicorn --bind 0.0.0.0:<PORT> app:app

# or

waitress-serve --listen=0.0.0.0:<PORT> app:app
```
Gunicorn or waitress can be installed via:

```bash
pip install gunicorn

# or

pip install waitress
```

At this point, you can test the app with existing training/validation data that you used in 
Milestone 2, and ping the app directly using the Python requests library, eg: 

```Python
X = get_input_features_df()
r = requests.post("http://0.0.0.0:<PORT>/predict", json=json.loads(X.to_json()))
print(r.json())
```

## Environments

The first thing you should setup is your isolated Python environment.
You can manage your environments through either Conda or pip.
Both ways are valid, just make sure you understand the method you choose for your system.
It's best if everyone on your team agrees on the same method, or you will have to maintain both environment files!
Instructions are provided for both methods.

**Note**: If you are having trouble rendering interactive plotly figures and you're using the pip + virtualenv method, try using Conda instead.

### Conda 

**Note: it is better to stick with pip environments in Docker containers!**

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate ift6758-conda-env

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-conda-env

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt



