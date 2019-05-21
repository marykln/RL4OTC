# RL4OTC
Exploring RL-Algorithms like A2C for a possible Application on the Unity Obstacle Tower Challenge 


# Approaches so far: 

## Actor-Critic: 
Implemented Python-Notebook based on the Actor-Critic Implementation found here (Actor Critic 4 CartPole): https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/8_Actor_Critic_Advantage 
### Issues: 
Struggled with OOM-Error - Ressource Exhausted on my Windows Environment 

## OpenAI Baselines 
### Idea: 
Use OpenAI Baselines for our RL-Algorithm
* First Idea: A2C 
* Possible Next approaches: ACKTR (?) 
https://openai.com/blog/baselines-acktr-a2c/

### Issues:
Couldn't install Baselines on my Windows Environment during some Package Errors 
#### Finally a Solution!! 
Set up Ubuntu Subsystem on Windows and reinstalled everything, so now the Baseline runs for their Gym-
Environment Pong-NoFrameskip-v4 

## Next steps: 
Exchange environment which is the input of the A2C Baseline with our Obstacle Tower Env 

#### Comparison: 
	
|  Environment Id    | Observation Space | Action Space                  | Reward Range | tStepL | Trials | rThresh |
|--------------------|-------------------|-------------------------------|--------------|--------|--------|---------|
| PongNoFrameskip-v4 | Box(210, 160, 3)  | Discrete(6)                   | (-inf, inf)  | 400000 | 100    | None    |
| ObstacleTower-Env  | Box(168,168,3)    | MultiDiscrete([3 3 2 3]) = 54 | (-inf, inf)  |        |        |         |
|                    |                   |                               |              |        |        |         |

#### ToDo: 
* Add Environment for run.py in OpenAI Baselines 
* Exchange last Layer for adapting Action space 
* Get a new Techfak password to access the Citec GPU Clusters


## Further Outview: 
Take a look at rwightmans pytorch approachs, which leads the agent to solve upto 10 floors (6-7 in average): https://github.com/rwightman/obstacle-tower-pytorch-a2c-ppo 
or even his rainbow approach, which leads to solving 7-8 floors in avg:
https://github.com/rwightman/obstacle-tower-pytorch-rainbow
