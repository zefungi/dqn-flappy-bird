import cv2
import os
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn

BATCH_SIZE = 32             # size of minibatch
LR = 1e-6                   # learning rate
INITIAL_EPSILON = 0.1       # starting value of epsilon
FINAL_EPSILON = 0.0001      # final value of epsilon

GAMMA = 0.99                # decay rate of past observations
TARGET_REPLACE_ITER = 100   # target net update frequency
MEMORY_CAPACITY = 50000     # number of previous transitions to remember

GAME = 'bird'               # the name of the game being played for log files
ACTIONS = 2                 # number of valid actions

OBSERVE = 1000.             # timesteps to observe before training, must be greater than BATCH_SIZE
EXPLORE = 200000.            # frames over which to anneal epsilon
EPOCH = 1000000             # number of game frames to train for
FRAME_PER_ACTION = 1        # how many frames to update the actions

TOTAL_REWARD = 0            # initial total reward

resize_w = 80
resize_h = 80

# 如果有 CUDA 可用，將 tensor 移動到 CUDA 上，否則保持在 CPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (resize_w, resize_h)), cv2.COLOR_BGR2GRAY)
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return observation


class Net_CNN(nn.Module):
    def __init__(self,):
        super(Net_CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600,256),
            nn.ReLU()
        )
        self.out = nn.Linear(256,2)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.to(next(self.parameters()).device)    # x.shape = (1, 4, 80, 80)
        x = self.conv1(x)                               # x.shape = (1, 32, 10, 10)
        x = self.conv2(x)                               # x.shape = (1, 64, 5, 5)
        x = self.conv3(x)                               # x.shape = (1, 64, 5, 5)
        x = x.view(x.size(0),-1)                        # x.shape = (1, 1600)
        x = self.fc1(x)                                 # x.shape = (1, 256)
        return self.out(x)

class DQN(object):
    def __init__(self, actions_num, observation):
        self.Q_eval_net, self.Q_target_net = Net_CNN().to(device), Net_CNN().to(device)
        self.memory = deque() 
        self.optimizer = torch.optim.Adam(self.Q_eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        
        # init parameters
        # timeStep = learn step conuter
        self.timeStep = 0   
        self.epsilon = INITIAL_EPSILON
        self.actions_num = actions_num
        self.load()

        # 使用四張遊戲剛開始的畫面（相同的圖片）作為初始狀態
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)

    def save(self):
        print("save model param")
        torch.save(self.Q_eval_net.state_dict(), 'params.pth')

    def load(self):
        if os.path.exists("params.pth"):
            print("load model param")
            self.Q_eval_net.load_state_dict(torch.load('params.pth'))
            self.Q_target_net.load_state_dict(torch.load('params.pth'))

    def choose_action(self):
        action = np.zeros(self.actions_num)

        # 根據遊戲畫面（當前狀態）選擇行為
        currentState = torch.Tensor(np.array([self.currentState])).to(device)
        QValue = self.Q_eval_net(currentState)[0]
        if QValue.device.type == 'cuda':
            QValue = QValue.cpu().detach().numpy()
        else:
            QValue = QValue.detach().numpy()

        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:     
                # random 
                action_index = random.randrange(self.actions_num)
                print("choose random action " + str(action_index))
                action[action_index] = 1
            else:
                # greedy
                action_index = np.argmax(QValue)
                print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # epsilon decay
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def store_transition(self, s, a, r, s_, terminal):
        # store the transition in self.memory
        self.memory.append((s, a, r, s_, terminal))

        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()

    def train(self): 
        # Step 0: 檢驗是否要將 target net 更新成 eval net
        if self.timeStep % TARGET_REPLACE_ITER == 0:
            self.Q_target_net.load_state_dict(self.Q_eval_net.state_dict())
            self.save()

        # Step 1: 從記憶庫中隨機取樣 minibatch(過去做過得動作，狀態，獎勵，下一個狀態，是否結束)
        minibatch = random.sample(self.memory, BATCH_SIZE)
        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        nextState_batch = np.array([data[3] for data in minibatch])
        
        action_index = action_batch.argmax(axis=1)
        nextState_batch = torch.Tensor(nextState_batch).to(device)
        print("action " + str(action_index))

        # Step 2: calculate y
        # 計算 Q target value （實際值）
        # 2-1. 計算下一個狀態所有動作的 Q 值
        QValue_batch = self.Q_target_net(nextState_batch).cpu().detach().numpy()

        # 2-2. 計算實際值 y = reward + gamma * 未来（下一個動作）預期最大值
        y_batch = np.zeros([BATCH_SIZE,1])
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(QValue_batch[i])
        y_target= Variable(torch.Tensor(y_batch).to(device))

        # 2-3. 計算預測值
        state_batch_tensor = Variable(torch.Tensor(state_batch).to(device))
        action_index = np.reshape(action_index,[BATCH_SIZE,1])
        action_batch_tensor = torch.LongTensor(action_index).to(device)
        y_predict = self.Q_eval_net(state_batch_tensor).gather(1, action_batch_tensor)

        # 2-4. 計算 loss
        loss = self.loss_func(y_predict, y_target)
        print("loss is "+str(loss.item()))

        # 2-5. backpropagation 更新參數
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, nextObservation, action, reward, terminal):
        # 捨棄最早的一張圖片，將新的圖片加入
        newState = np.append(self.currentState[1:], [nextObservation], axis = 0)
        
        # 儲存記憶到記憶庫
        self.store_transition(self.currentState, action, reward, newState, terminal)

        # Train the network
        if self.timeStep > OBSERVE: 
            self.train()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        else:
            state = "train"
        print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

        # 更新當前狀態
        self.currentState = newState
        self.timeStep += 1

if __name__ == '__main__': 
    # Step 1: init Flappy Bird Game (with pygame)
    flappyBird = game.GameState() 

    # Step 2: play game
    # Step 2.1: obtain init state
    action_init = np.array([1,0]) # do nothing
    observation_init, reward0, terminal = flappyBird.frame_step(action_init)
    observation_init = img_preprocess(observation_init)
    
    # Step 3: init DQN
    dqn = DQN(actions_num = 2, observation = observation_init) 

    # Step 3.2: run the game
    for i in range(EPOCH):
        action = dqn.choose_action()                                        # 根據當前狀態選擇行為

        nextObservation, reward, terminal = flappyBird.frame_step(action)   # 根據選擇的行為得到新的反饋
        nextObservation = img_preprocess(nextObservation)                   

        dqn.update(nextObservation, action, reward, terminal)               # 儲存當前狀態、行為、反饋、新狀態、是否結束
                                                                            # 訓練網路
                                                                            # 更新當前狀態、時間步數
        TOTAL_REWARD += reward
    
    printz
