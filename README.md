# KARlab Use Case 3: Unity (ML-Agents) Proof of Concept of an Autonomous Parking System with Reinforcement Learning [Lucas Steffanutto]
 

## Notes for beginners
- The ML-Agents installation and version used is avalaible [here](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2257420297/ML-Agents+Toolkit+Documentation)
- [Lucas Steffanutto] I made this code with the help of the following tutorials:
- Based on the [idea/realisation of Dilmer Valecillos] https://github.com/dilmerv/UnityMLEssentials
- Well modelize the vehicle [WheelCollider Unity Tutorial](https://docs.unity3d.com/Manual/WheelColliderTutorial.html)
- [First ML-Agent Unity Tutorial](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Create-New.md)
- Script to understand for continuous actions: [3D Ball ML-Agents example](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#3dball-3d-balance-ball)
- The others [ML-Agents example](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) are also essentials to understand in order to developp your own model.
- **The code of the POC 1** is in: "Assets\ML-Agents\Examples\CarAgentSimple\Scripts"
- **The Final Code with Scenario 2** is in: "Assets\ML-Agents\Examples\CarAgentScenario2
\Scripts"
- To train the Agent with other Configuration File (.yaml), go to: ml-agents-release_17\config\ppo and then create a text file ".yaml" as the other, writing under "behavior" the name of the script associated to the agent.
- Asset to download for the kart visual of this project [Steampunk Kart](https://assetstore.unity.com/packages/3d/vehicles/land/steam-punk-kart-58941) (in order to complete " Assets/SteamPunk_Kart/Textures/* " of the .gitignore)
- Download also "Time_Starter.zip" of this [Unity time tutorial](https://learn.unity.com/tutorial/time-0fbw?uv=2019.4&courseId=5dd851beedbc2a1bf7b72bed&projectId=5df2611eedbc2a0020d90217#), execute it in a Unity Editor in order to recuperate the "Adam Character Pack" and complete the files of the repository "Assets\ML-Agents\Examples\CarAgentScenario2\Scripts\AdamCharacterPack\Adam" (after pulling this repo) 
- After running the "Doxyfile" in "doxygen/Doxyfile" with Doxywizard, the Doxygene of this project will be available in "doxygen/html/index.html" > files > files list > CarAgentS2.cs. See [DoxySetup](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/1626374252/Code+documentation+with+Doxygen#Setup-Doxygen) for more informations.
## Actual PPO Developpement of the POC
- We can select the logs that we want to see in the console, in the Unity Editor: car_root>Inspector>Car Agent "Show xxx Logs"
- When we developp a feature of the Kart modelisation, it is important to test it using the keyboard and choosing Heuristic Only in Behavior Parameters>Behavior Type in the agent inspector
- The appartition of the localisation of the parking target and the agent is random. The orientation of the agent is also random.
- You can add randomize the apparitions of obstacles/walker and the kart Agent
- You can add 2 other kart to train the Agent to Parking evitating the others kart (voir Scenario 2 avec 3 karts ci-dessous)
- Reward Function for the agent for each timestep ** t **: (-1) if the agent collides an obstacle (kart,human,limit or block), if is more than 50m of the parking target, if reach off the MaxSteps=3000, or if his parking alignement with the parking target has an error >10°; (+1) if he is stopping >1seconde on the parking target with an alignement <±10°;  (-1/MaxStep) if the agent chose to backward (negative Torque) or turn >15°; (-1f / MaxStep) * (distanceToTarget) each timestep (only during training, get it off when the agent seems do a good parking, in order to see if the Mean Reward of the Agent is near of 1 when he is well trained) 
### Scenario 1
- INPUT = State Space = Vehicle observation = the vehicle position, his heading angle (from z axis), his velocity from the two plan axes and the parking target position (Space size = 9)
- OUTPUT = Wheels Steering Angle and Motor Torque (Continuous Actions)
### Scenario 2
- INPUT 1 = State Space = Vehicle observation = his heading angle, his velocity from the two plan axes (x and z), the parking target position, the 3D vector of the remaining distance (Space size = 12)
- INPUT 2 = State Space = Vehicle Radar observation = 9 rayscast with 4 tags, "bloc", "walker", "limit" and "kart" (Space size = 54)
- OUTPUT = Wheels Steering Angle, Motor Torque and Brake (Continuous Actions)
- ** See directly [Modélisation pratique et implémentation du Stationnement sur Unity on Confluence](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2139717760/Mod+lisation+pratique+et+impl+mentation+du+Stationnement+Autonome+sur+Unity) for more details **

### Scenario 2 avec 3 karts
- Load the following scene: Assets\ML-Agents\Examples\CarAgentScenario2\SCENARIO_2_3_Kart et activer l'environnement "SCENARIO_2_3_KartMK" and put "true" for parkingZoneEnabled (ligne 84) to train the agent with 2 other kart who are parking + write "3" and "5" in the inspector field "ParkingZone" for each of the two others karts
- Plus de détails et résultats [ici](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2415231261/Agent+Training+Tab+Summary#Final-Tests-Scenario-2-avec-3-kart-with-PPO-et-Curriculum-Learning-et-BC%2BGail%2BCuriosity)

### The following Logs are avalaible
- We have the position of the vehicle and of the parking target
- We have the orientation/heading angle of the vehicle and of the parking target and their relative orientation
- We have the Total Reward through the episode and the reward of the episode
- The Agent TimeStep and time of an episode, of the total
- The same with the Academy
- The radar logs with the obstacle detected and its distance to the agent
- The collision with and object

## Training
We use the following references to test different configurations
- [State of the Art Resume for Autonomous Parking](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2107604999/R+sum+des+papiers+pr+c+dents+sur+l+tat+de+l+art+APS)
- [ML-Agent Training Documentation](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2269839379/Documentation+ML-Agent+Release+17#Design-de-l%E2%80%99entra%C3%AEnement-%5B3%5D)
- In the scenario 2 for each episode, we randomize the position of the parking target, the init of the kart Agent and obstacles
- To train the Unity Agent, Launch from the terminal: [(mlagents-r17-OUI)>...\ml_agents_release_17] $ " mlagents-learn Config\KARlab2_RollerBall.yaml --run-id=MyTrainingDemo "
- To visualize the results: " tensorboard --logdir results --port 6006 " then open a browser: " http://localhost:6006/ "
- Start a training from a previous good model/training: add the option " --initialize-from=PPO_OK_scenario1 " at the end of the mlagents-learn" training command
- Resume a training from a previous stopped model/training: add the option " --resume " before "--run-id=MyTrainingDemo"
- The details of Curriculum Learning and BC+GAIL+Curiosity are avalaible [here]
- The configuration files are avalaible here: \\s-gaia\Avisto\Projets\000000872_-_KARlab\Résultats_UC3\Agent_Training\results

## Best Results and Configurations obtained
- **Good Results of the POC 1** is in: "Assets\ML-Agents\Examples\CarAgentSimple\good_results"
- **Good Configs of the POC 1** is in: "Assets\ML-Agents\Examples\CarAgentSimple\good_configs"
- The **Final Scenario 1 & 2** results are in: " Assets\ML-Agents\Examples\CarAgentScenario1\Config\results " & " Assets\ML-Agents\Examples\CarAgentScenario1\Config\results " 
## Comparison of the configurations
- Training Tab [here](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2415231261/Agent+Training+Tab+Summary)
- All the interesting Test and training are available on Gaïa: "\\s-gaia\Avisto\Projets\000000872_-_KARlab\Résultats_UC3\"
- The results are [here](https://advans-group.atlassian.net/wiki/spaces/KARlab/pages/2415231261/Agent+Training+Tab+Summary#Inf%C3%A9rences-Resume)
- Final RESULT: [INFERENCE](https://drive.google.com/file/d/1MLnOcGO_hjnwQfAxk59FrPND0IrpUXIS/view?usp=sharing)

