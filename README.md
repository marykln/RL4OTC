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

1. Select Floor with Key on the ground level: 
* select a floor where a key is located to start with: Floor 10
* specifiy tower seed where key is nearby: Tower (Seed) 0 
	* Second room holds the key on the ground level ( so no jumping needed) .
* start with a stack of hard coded actions to get into the room where the key is located and see the key 

2. Decrease Action-Space:  
* create action dictionary for all possible actions {1:[1,0,0,0] = just go forward, 2:[1,1,0,0]=go forward and rotate and so on} decrease 54 possible actions to action 
	* 1 move forward, 
	* action 2 turn camera rotation left, 
	* action 3 turn camera rotation right and 
	* 4 jump forward 

3. Curriculum Learning:
* start with backward experience learning so: 
	* handcrafted way to the key (sequence) and than the next time only handcraft the sequence nearby the key and for every new reset make the distance to the key greater to learn a huger sequence 
* Exchange environment which is the input of the A2C Baseline with our Obstacle Tower Env 

4. Implement own Reward-Function: 
* Write own reward function that checks after every event step if key is picked up by checking the discrete box and gets a reward if value > 0 (1 + 1, like for 10 steps: 0 0 0 0 0 0 0 0 0 1 )


##### Later: 
than continue with a floor where the key is on a stair 
when it founds the key only select the parts right before finding the key plus a part of the same size to use it for training (collect experiences, for example 10.000 )
after that improve to more complicated floors like with key on stairs


#### Comparison: 
	
|  Environment Id    | Observation Space | Action Space                  | Reward Range | tStepL | Trials | rThresh |
|--------------------|-------------------|-------------------------------|--------------|--------|--------|---------|
| PongNoFrameskip-v4 | Box(210, 160, 3)  | Discrete(6)                   | (-inf, inf)  | 400000 | 100    | None    |
| ObstacleTower-Env  | Box(168,168,3)    | MultiDiscrete([3 3 2 3]) = 54 | (-inf, inf)  | 1000   |        |         |
|                    |                   |                               |              |        |        |         |

#### ToDo: 
* Add Environment for run.py in OpenAI Baselines 
* Exchange last Layer for adapting Action space 
* Get a new Techfak password to access the Citec GPU Clusters


## Further Outview: 
Take a look at rwightmans pytorch approachs, which leads the agent to solve upto 10 floors (6-7 in average): https://github.com/rwightman/obstacle-tower-pytorch-a2c-ppo 
or even his rainbow approach, which leads to solving 7-8 floors in avg:
https://github.com/rwightman/obstacle-tower-pytorch-rainbow
