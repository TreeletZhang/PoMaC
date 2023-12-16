import numpy as np
from multiagent_i2c.core import World, Agent, Landmark
from multiagent_i2c.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 5
        num_landmarks = 5
        world.collaborative = True 
        world.discrete_action = True
        world.num_agents_obs = 4   
        world.num_landmarks_obs = 4  
        
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
       
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        
        self.reset_world(world)
        return world

    def reset_world(self, world):
        
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        
        for agent in world.agents:
            # treelet
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            # if i != 0:
            #     for j in range(i):
            #         while True:
            #             if np.sqrt(np.sum(np.square(landmark.state.p_pos - world.landmarks[j].state.p_pos)))>0.22:
            #                 break
            #             else: landmark.state.p_pos = np.random.uniform(-world.range_p, +world.range_p, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            collision_dist = agent.size + l.size
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < collision_dist:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        collisions += 0.5
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collision_dist = agent1.size + agent2.size
        return True if dist < collision_dist else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # local reward
        #dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        #rew = rew - min(dists)
        # global reward
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            #rew -= min(dists)
            # treelet
            # rew -= min(dists)/len(world.agents)
            rew -= min(dists)
            # treelet
        
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)
        #
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)
        # 出界惩罚
        # for a in world.agents:
        #     if (a.state.p_pos[0]<-1) or a.state.p_pos[0]>1:
        #         rew -= 1
        #     if (a.state.p_pos[1] < -1) or a.state.p_pos[1] > 1:
        #         rew -= 1

        return rew
        #return rew*0.025  # treelet

    def observation(self, agent, world):
        # get positions of predefined landmarks
        entity_pos = []
        dis_lm_n = []
        num_landmarks_obs = world.num_landmarks_obs
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            dis_lm_n.append(np.sqrt(np.sum(np.square(agent.state.p_pos - entity.state.p_pos))))  
        sort_index = sorted(range(len(dis_lm_n)), key=lambda k: dis_lm_n[k])
        near_lm_pos = [entity_pos[sort_index[i]] for i in range(num_landmarks_obs)]  
        # get positions of predefined agents
        other_pos = []
        dis_agent_n = []
        num_agents_obs = world.num_agents_obs
        for other in world.agents:
            if other is agent: continue
            dis_agent_n.append(np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos))))
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        sort_index = sorted(range(len(dis_agent_n)), key=lambda k: dis_agent_n[k])
        near_agent_pos = [other_pos[sort_index[i]] for i in range(num_agents_obs)]
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + near_lm_pos + near_agent_pos)  


