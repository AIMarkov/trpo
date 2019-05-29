#pytorch 0.4
#思路,虽然单调递增,但是学习速度很慢,因此可以通过引导来加速其学习速率
#实际上这使用的是高斯策略

import argparse
from itertools import count

import pybullet_envs
import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import csv

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
out=open("AntBulletEnv.csv",'a',newline="")
csv_write=csv.writer(out,dialect="excel")
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="AntBulletEnv-v0", metavar='G',
                    help='name of the environment to run')#HumanoidStandup
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',default=1,help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
print('num_input:',num_inputs)
num_actions = env.action_space.shape[0]
print('num_actions:',num_actions)

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)


def save_models():
    torch.save(policy_net.state_dict(), 'models/best_Policy.model')
    torch.save(value_net.state_dict(),'models/best_Value.model')
def load_G_and_D(self):
    self.G.load_state_dict(
        torch.load('models/best_Policy.model')
    )


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):

    rewards = torch.Tensor(batch.reward)
    #print('rewards:', rewards)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))#数组拼接
    #print('batch.action:', batch.action)
    #print('actions:', actions)
    states = torch.Tensor(batch.state)
    #print('states:', states)
    values = value_net(Variable(states))
    #print('values:', values)

    returns = torch.Tensor(actions.size(0),1)
    #print('returns:', returns)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)


    prev_return = 0#因为memeroy中一定是包含完整的情节，而会只包含某个情节的一部分,所以这三个值一定会是从０开始累计的
    prev_value = 0
    prev_advantage = 0
    #print('reward.shape():',rewards.size(0))#奖赏还是一维,不能使用shape
    for i in reversed(range(rewards.size(0))):#mask[i]=0,mask[i]=0这个标志，完美避免情节之间的干扰，这是因为无论是return,delta,advantage,都是从后面往前面求得，当前的值求取与其前面的值是无关的，只与其后的值有关
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]

        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]
    #exit()

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss,可能是pytorch中没有这个优化函数，为什么ＴＲＰＯ更好，还有一个重要原因就是他是ＭＣ的Value
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)#这个地方很有意思，网络的优化是使用的Scipy中的优化算法，注意scipy是对numpy进行操作的
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()#为甚么还要均一化

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

running_state = ZFilter((num_inputs,), clip=5)#num_inputs是状态输入size,hopper 11
running_reward = ZFilter((1,), demean=False, clip=10)

steps=0
for i_episode in count(1):
    memory = Memory()#这个经验池只存放单情节经验,不应该是单批量经验

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()
        #print('state1:',state)
        state = running_state(state)
        #print('state2:', state)
        #exit()

        reward_sum = 0
        for t in range(1000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            steps+=1
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)
            # if t>=3:
            #     break
            #print("render:",args.render)
            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    #print("memory:",len(memory.memory))
    batch = memory.sample()
    update_params(batch)
    save_models()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}\tsteps {}'.format(
            i_episode, reward_sum, reward_batch,steps))
        csv_write.writerow([reward_sum,steps])
