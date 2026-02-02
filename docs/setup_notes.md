# Setup Notes

## CARLA Setup
This codebase is developed on CARLA 0.9.10.1 with Unreal Engine 4.24\
Download link: https://github.com/carla-simulator/carla/releases/tag/0.9.10.1 \
Documentation link: https://carla.readthedocs.io/en/0.9.10/ \
Docker hub link: https://hub.docker.com/r/carlasim/carla/tags 

## Environment Setup

Environment has to be setup based on the requirements of the driving agents
as well as the attacks and defenses.
For older GPUs, cuda 10.2 is available as precompiled binary. Newer GPUs 
require dependencies to be resolved appropriately.

#### enviroment setup for transfuser with cuda 11.3 (for newer gpus cuda 10.2 is not available as precompiled binary)
```
conda create -n tfuse python=3.7
conda activate tfuse
conda install cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard matplotlib ipykernel ipython ipywidgets jupyter-client jupyter-core notebook
pip install tqdm tabulate timm==0.5.4 scikit-image scikit-learn scipy pytorch-lightning
pip install pygame pandas opencv-python
pip install imgaug
pip install open3d
pip install py-trees==0.8.3
pip install ephem
pip install dictor
pip install mmsegmentation==0.25.0 mmdet==2.25
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install ujson
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install numba
```

If while installing torch-scatter, manual installation from source has to be
avoided, then install via ```pip install --no-index```.

#### environment setup for interfuser
Dependencies listed in Interfuser's requirements.txt needs to be installed. 
If environment for transfuser is already created then you may only need to 
install ```easydict```. Additionally, in case of timm related errors, 
first try to use the timm folder included in the interfuser repository by
modifying the import statements in the dependent python files accordingly. 

In addition to installing the requirements for transfuser and interfuser agent, few other libraries are needed to be installed such as adversarial robustness toolbox, torchinfo, etc.

Our conda environment details have also been included here in docs/

## Execution Notes

### Start CARLA server
Command line options for Unreal Engine 4:\
https://dev.epicgames.com/documentation/en-us/unreal-engine/command-line-arguments?application_version=4.27
/
#### start carla server with nvidia gpu 
```
./CarlaUE4.sh -prefernvidia
```
#### start carla server without display
As per https://carla.readthedocs.io/en/0.9.10/adv_rendering_options/
```
DISPLAY= ./CarlaUE4.sh -opengl
```
But, this does not work.

Try this instead:
```
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -opengl
```

#### carla server using docker container without display
```
sudo docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.10.1 /bin/bash -c 'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh'
```

#### start carla server with defined resolution

```
./CarlaUE4.sh --world-port=2000 -Windowed -ResX=1920 -ResY=1080
```
Windowed, ResX, ResY are Unreal Engine commandline arguments. For full list of
Unreal Engine commandline arguments check here: https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-command-line-arguments-reference (Ensure list is for the 
version of Unreal Engine used in the respective Carla version.)

### Run experiment locally
```
experiments/scripts/local_evaluation.sh <experiment config file>
```
For example,
```
experiments/scripts/local_evaluation.sh experiments/scripts/configs/g7config
```

### Other setup notes
```pycm``` has dependency on a different art package than adversarial 
robustness toolbox (also 'art'). So, installation will overwrite.
Avoid using art dependent pycm calls, and use source code.