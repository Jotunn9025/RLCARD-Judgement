import argparse
import os
import torch
import numpy as np

from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import set_seed, tournament, reorganize
from judgement.env import JudgementEnv

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Training env (Self-play)
    env = JudgementEnv({
        'allow_step_back': False,
        'starting_set_cards': args.cards,
    })
    
    # Evaluation env (Agent vs 3 Randoms)
    eval_env = JudgementEnv({
        'allow_step_back': False,
        'starting_set_cards': args.cards,
    })

    agents = []
    for i in range(env.num_players):
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[i],
            hidden_layers_sizes=[256, 256],    
            q_mlp_layers=[256, 256],
            anticipatory_param=0.1,
            batch_size=256,
            rl_learning_rate=1e-4,             
            sl_learning_rate=args.sl_lr,
            min_buffer_size_to_learn=2000,     
            q_replay_memory_init_size=2000,
            q_replay_memory_size=100000,       
            reservoir_buffer_capacity=100000,
            device=device
        )
        agents.append(agent)

    env.set_agents(agents)
    random_agent = RandomAgent(num_actions=env.num_actions)

    print(f"Training on {device} for {args.episodes} episodes...")

    # 3. Training Loop
    for episode in range(args.episodes):
        
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        for i in range(env.num_players):
            for ts in trajectories[i]:
                agents[i].feed(ts)

        if episode % args.evaluate_every == 0:
            # Evaluate Agent 0 against 3 Random Agents
            eval_env.set_agents([agents[0], random_agent, random_agent, random_agent])
            rewards = tournament(eval_env, args.evaluate_num)
            rl_loss = getattr(agents[0], 'rl_loss', 0)
            sl_loss = getattr(agents[0], 'sl_loss', 0)
            
            print(f"Episode: {episode}")
            print(f"  >> Payoff vs Random: {rewards[0]:.3f}")
            print(f"  >> Avg Payoff (Self-Play): {np.mean(payoffs):.3f}")
            if rl_loss: print(f"  >> RL-Loss: {rl_loss:.4f} | SL-Loss: {sl_loss:.4f}")
            print("-" * 40)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    agents[0].save_checkpoint(args.save_dir, filename='best_agent_13cards.pth')
    print(f"Training complete. Model saved to {args.save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("NFSP training in Judgement Env")
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--evaluate_every', type=int, default=500)
    parser.add_argument('--evaluate_num', type=int, default=100)
    parser.add_argument('--cards', type=int, default=13)
    parser.add_argument('--sl_lr', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='nfsp_checkpoints')

    args = parser.parse_args()
    train(args)