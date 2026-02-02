## Changes for AV Safety

### data folder
* Adds longest6 benchmark (from transfuser, but checked with carla_garage for differences), original README inside folder
* Adds town_maps_tga (in carla_garage, maps in transfuser), town_maps_xodr
* Adds transfuser++ training routes and scenarios under single folder transfuser
TODO: Add other benchmarks and splits from carla garage and elsewhere

### leaderboard folder
* run_step does not have sensors param, and accordingly other files from carla_garage are also modified. Follows default leaderboard autonomous_agent run_step definition.
* no changes are made to original files, only carla_garage added files which end with _local.
* envs in carla_garage has few commented print statements, not added here.
* route_scenario_local from carla_garage
    * has some benchmark specific initialization in _initialize_actors
    * adds customizable BLOCKED_THRESHOLD for blocked_criterion evaluation
* route_scenario.py, line 459: Town10 is renamed as Town10HD in carla_garage.
* Added changes from statistics_manager.py to statistics_manager_local.py
* route_indexer.py and route_parser.py from carla_garage are added here with _local in their names as there are modifications. Though they may not be required. Updated leaderboard codes have routes_subset param. (_local files were removed to cleanup)
    * Important: Use route_indexer.py and route_parser.py as it includes routes_subset.
* Added leaderboard_evaluator_local.py with necessary modifications to support new route subsets features and changes in statistics manager. Also checked with leaderboard_evaluator_local_adversarial.py from attack-transfuser.
* autonomous_agent_local.py not being used - deleted. Changes in autonomous_agent.py itself.

### scripts folder
* have new files make_docker.sh, generate_evalai_results.py, generate_evalai_stdout.py, and merge_statistics.py
* copied manage_scenarios.py from carla_garage
    * purpose similar to that of set_new_scenario.py, but different code. need to check.
* scritps for running my experiments moved to separate experiments folder outside.

## Original CHANGELOG
## Latest changes

* Added a new attribute to the global statistics, *scores_std_dev*, which calculates the standard deviation of the scores done throughout the simulation.
* Fixed bug causing the global infractions to not be correctly calculated

* Creating stable version for the CARLA online leaderboard
* Initial creation of the repository
