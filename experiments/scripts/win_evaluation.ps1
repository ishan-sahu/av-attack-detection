Param(
    [Parameter(Mandatory, HelpMessage="Please provide a valid config file path!")]
    [string]$configPath
)
. $configPath
Write-Host "Experiment Config: $configPath"
Write-Host "CARLA Root: $env:CARLA_ROOT"
Write-Host "Working Directory: $env:WORK_DIR"
Write-Host "CARLA Server: $env:MYHOST:$env:MYPORT"
Write-Host "Server Working Directory: $env:SERVER_WORK_DIR"

#$env:PYTHONIOENCODING="utf-8"
#$env:PYTHONLEGACYWINDOWSSTDIO="utf-8"

$CARLA_SERVER = "$env:CARLA_ROOT/CarlaUE4.exe"
$env:PYTHONPATH += ";$env:CARLA_ROOT/PythonAPI"
$env:PYTHONPATH += ";$env:CARLA_ROOT/PythonAPI/carla"
$env:PYTHONPATH += ";$env:CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
$SCENARIO_RUNNER_ROOT = "$env:WORK_DIR/scenario_runner"
$LEADERBOARD_ROOT = "$env:WORK_DIR/leaderboard"
$env:PYTHONPATH += ";$env:CARLA_ROOT/PythonAPI/carla/;$SCENARIO_RUNNER_ROOT;$LEADERBOARD_ROOT"

$ROUTES = "$env:WORK_DIR/leaderboard/data/$_ROUTES_PATH"
$SCENARIOS = "$env:WORK_DIR/leaderboard/data/$_SCENARIOS_PATH"
Write-Host "Route: $ROUTES"
Write-Host "Evaluation Scenarios: $SCENARIOS"


$TEAM_AGENT = "$env:WORK_DIR/$_TEAM_AGENT"
$TEAM_CONFIG = "$env:WORK_DIR/$_TEAM_MODEL"
Write-Host "Driving Agent: $TEAM_AGENT"
Write-Host "Driving Model: $TEAM_CONFIG"

$CHECKPOINT_ENDPOINT = "$env:WORK_DIR/$_CHECKPOINT_PATH"
$RECORD_PATH = "$env:SERVER_WORK_DIR/$_RECORD_PATH"
Write-Host "Checkpoint Save Path: $CHECKPOINT_ENDPOINT"
Write-Host "Recording Save Path: $RECORD_PATH"

$REPETITIONS = $_REPETITIONS
$CHALLENGE_TRACK_CODENAME = $_CHALLENGE_TRACK_CODENAME
$DEBUG_CHALLENGE = $_DEBUG_CHALLENGE
#$DEBUG_CHALLENGE=1
# testing code right now, therefore no resume; for complete evaluation
# runs set resume to true
#$RESUME=1
$RESUME = $_RESUME
$env:DATAGEN = $_DATAGEN # some code is trying to access this as os env variable
Write-Host "Repetitions: $REPETITIONS"
Write-Host "Track: $CHALLENGE_TRACK_CODENAME"
Write-Host "Challenge Debug Mode: $DEBUG_CHALLENGE"
Write-Host "Resume: $RESUME"
Write-Host "Datagen Mode: $env:DATAGEN"

# python3 goes to windows app, included in windows path variable 
python "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_local.py" `
--scenarios=$SCENARIOS  `
--routes=$ROUTES `
--repetitions=$REPETITIONS `
--track=$CHALLENGE_TRACK_CODENAME `
--checkpoint=$CHECKPOINT_ENDPOINT `
--agent=$TEAM_AGENT `
--agent-config=$TEAM_CONFIG `
--debug=$DEBUG_CHALLENGE `
--resume=$RESUME `
--host=$env:MYHOST `
--port=$env:MYPORT `
--record=$RECORD_PATH
