## Config files for experiments

### Defining experiment config file
Need to define following variables
* CARLA_ROOT - root directory where carla is installed
* WORK_DIR - working directory / base directory of the av-safey codebase
* SERVER_WORK_DIR - if the carla is running on a different machine, specify the directory where the recordings will be saved. In case carla and evaluation experiment are running on the same machine, this can be set the same as WORK_DIR.
* MYHOST - host ip of the carla server, in case same machine for carla server and experiment, ```localhost``` can be used.
* MYPORT - port of the carla server
* _TEAM_AGENT - Path to the file where autonomous agent is defined. In case of adversarial experiment, path to the adversarial autonomous agent wrapper.
* _TEAM_MODEL - Path to the pretrained autonomous agent model. In case of adversarial experiment, path to the adversarial attack config file.
* _ROUTES_PATH - Path to the routes file. For evaluation experiments, we would be using ```longest6/longest6_split/longest_weathers_0.xml``` from leaderboard 1.0 included in this codebase.
* _SCENARIOS_PATH - Path to the file with scenarios for evaluation.
* _CHECKPOINT_PATH - File path where results statistics for the experiment is to be saved.
* _RECORD_PATH - Path where the recording of the experiment is to be saved
* _REPETITIONS - Number of times the experiment is to be repeated
* _CHALLENGE_TRACK_CODENAME - Track of the CARLA challenge under which evaluation is to be done. At present, only SENSORS is supported.
* _DEBUG_CHALLENGE - whether to run in debug mode.
* _RESUME - whether to resume from previous experiment's last executed RouteScenario
* _DATAGEN - whether to run in data generation mode, and not in evaluation mode. This option is not used in our codebase, keep it 0.

    For example for driver agent evaluation,
    ```
    CARLA_ROOT="/home/ishan/dev/carla/CARLA_0.9.10.1"
    WORK_DIR="/home/ishan/dev/repos/github/av-attack-detection"
    SERVER_WORK_DIR="/home/ishan/dev/repos/github/av-attack-detection"
    MYHOST="localhost"
    MYPORT="2000"
    _TEAM_AGENT="drivers/team_code_transfuser/submission_agent.py"
    _TEAM_MODEL="/drivers/pretrained_models/transfuser"
    _ROUTES_PATH="longest6/longest6_split/longest_weathers_0.xml"
    _SCENARIOS_PATH="longest6/eval_scenarios.json"
    _CHECKPOINT_PATH="results/results_transfuser_longest6_1.json"
    _RECORD_PATH="recordings/"
    _REPETITIONS=1
    _CHALLENGE_TRACK_CODENAME=SENSORS
    _DEBUG_CHALLENGE=0
    _RESUME=0
    _DATAGEN=0
    ```

### Defining adversarial attack config file
In case of evaluation of the driving agent under attack, this config file is required. For driving agent evaluation under normal conditions with no attack, this is not required. This is in the form of a json file.

There are few common variables across attacks and few attack specific variables.

Common variables:

* attack: Name of the attack - poltergeist, snal, or esia
* agent: Path to the file where autonomous agent is defined. This agent would be evaluated under attack.
* agent_config: Path to autonomous agent config file
* num_skip_frames: Number of frames to skip between two consecutive attacks. With reference to ```d``` defined in the paper, ```num_skip_frames = d-1```. For example, if you want to generate results for ```d = 1```, set num_skip_frames as 0.
* disp_interface: This is not being used. Keep it 0.

Attack specific variables:

Poltergeist:

* datagen_path: This is used when generating dataset for detector model training and evaluation. See dataset generating notes.

SNAL:

* eps_n: bound on maximum perturbation that the attacker can introduce in terms of l_inf norm. In our experiments we use 4, 8.
* candidates_list_file: Path to the pickle file in whic candidate objects are saved for attack.

ESIA:
 
* severity: severity of ESIA. 1 is mild, 2 is moderate, and 3 is severe.

Config files for different experiments are included in the codebase in experiments folder. Please change the respecitive paths carefully.
Attack config files are present inside experiments/scripts/configs/<agent_name>/<attack_name>
