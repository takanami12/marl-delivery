from env import Environment
#from agent import Agents
from greedyagent import GreedyAgents as Agents

import numpy as np

import json

def collect_sft_data(env: Environment,
                     agent_class,
                     n_episodes: int,
                     max_steps_per_ep: int,
                     output_path: str):
    """
    Chạy n_episodes, thu thập {prompt, response} từ agent greedy,
    và lưu vào output_path (JSONL).
    """
    data = []
    agents = agent_class()

    for ep in range(n_episodes):
        state = env.reset()
        agents.init_agents(state)
        for t in range(max_steps_per_ep):
            # 1) Tạo prompt từ state
            prompt = encode_state(state)  
            
            # 2) Lấy actions từ agent
            actions = agents.get_actions(state)
            # chuyển list of tuples thành text response
            response = format_actions(actions)
            
            # 3) Thêm vào data
            data.append({
                "prompt": prompt,
                "response": response
            })
            
            # 4) Bước tiếp
            state, reward, done, _ = env.step(actions)
            if done:
                break

    # Ghi ra file JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Collected {len(data)} samples into {output_path}")

# Ví dụ helper functions bạn cần tự định nghĩa:
def encode_state(state) -> str:
    # Serialize state thành text, ví dụ:
    # "Time: 0; Robots: [(x1,y1),(x2,y2),...]; Packages: [(locA,locB), ...]"
    print(state["robots"][1])
    robots_str = ";".join(f"R{i}:{pos}" for i,pos in enumerate(state["robots"]))
    pkgs_str   = ";".join(f"P{i}:{(s,d)}" for i,(s,d) in enumerate(state["packages"]))
    return f"Time: {state['time_step']} | {robots_str} | {pkgs_str}"

def format_actions(actions) -> str:
    # Chuyển list như [(move,0),(pick,3),...] thành chuỗi: "move 0; pick 3; ..."
    return "; ".join(f"{act} {idx}" for act,idx in actions)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map.txt", help="Map name")

    args = parser.parse_args()
    np.random.seed(args.seed)

    env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                      n_robots=args.num_agents, n_packages=args.n_packages,
                      seed = args.seed)
    
    state = env.reset()
    agents = Agents()
    agents.init_agents(state)
    print(state)
    #env.render()
    done = False
    t = 0

    collect_sft_data(env, Agents, n_episodes=100, max_steps_per_ep=50, output_path="sft_data.jsonl")
    while not done:
        actions = agents.get_actions(state)
        next_state, reward, done, infos = env.step(actions)
        state = next_state
        env.render()
        t += 1

    print("Episode finished")
    print("Total reward:", infos['total_reward'])
    print("Total time steps:", infos['total_time_steps'])
