#!/bin/bash
source $1
echo "Experiment Config: $1"
echo "CARLA Root: ${CARLA_ROOT}"
echo "Working Directory: ${WORK_DIR}"
echo "CARLA Server: ${MYHOST}:${MYPORT}"
echo "Server Working Directory: ${SERVER_WORK_DIR}"

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export ROUTES=${WORK_DIR}/leaderboard/data/${_ROUTES_PATH}
export SCENARIOS=${WORK_DIR}/leaderboard/data/${_SCENARIOS_PATH}
echo "Route: ${ROUTES}"
echo "Evaluation Scenarios: ${SCENARIOS}"

export TEAM_AGENT=${WORK_DIR}/${_TEAM_AGENT}
export TEAM_CONFIG=${WORK_DIR}/${_TEAM_MODEL}
echo "Driving Agent: ${TEAM_AGENT}"
echo "Driving Model: ${TEAM_CONFIG}" 

export CHECKPOINT_ENDPOINT=${WORK_DIR}/${_CHECKPOINT_PATH}
export RECORD_PATH=${SERVER_WORK_DIR}/${_RECORD_PATH}
echo "Checkpoint Save Path: ${CHECKPOINT_ENDPOINT}"
echo "Recording Save Path: ${RECORD_PATH}"

export REPETITIONS=${_REPETITIONS}
export CHALLENGE_TRACK_CODENAME=${_CHALLENGE_TRACK_CODENAME}
export DEBUG_CHALLENGE=${_DEBUG_CHALLENGE}
export RESUME=${_RESUME}
export DATAGEN=${_DATAGEN}
echo "Repetitions: ${REPETITIONS}"
echo "Track: ${CHALLENGE_TRACK_CODENAME}"
echo "Challenge Debug Mode: ${DEBUG_CHALLENGE}"
echo "Resume: ${RESUME}"
echo "Datagen Mode: ${DATAGEN}"

python3 -W ignore ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--host=${MYHOST} \
--port=${MYPORT} \
--record=${RECORD_PATH} \
--timeout=600.0
