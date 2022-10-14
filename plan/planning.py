import sys
sys.path.append("../reason")

from config import gen_args
from agent import Agent
from vis_planning import html

args = gen_args()

if __name__ == '__main__':
    for phase in ['valid', 'unseen', 'train']:
        for data_folder in ['plan-multi']:
            num_evaluation, total_success_rates = 50, [0, 0, 0]
            agent = Agent(args, phase, data_folder)
            for sample_idx in range(num_evaluation):
                print("Starting goal-conditioned manipulation for [{}]".format(sample_idx+1))
                success_rates = agent.run_agent(sample_idx)
                for i in range(len(success_rates)):
                    total_success_rates[i] += success_rates[i]

            for j in range(len(total_success_rates)):
                if j == 0:
                    version = 'graph'
                if j == 1:
                    version = 'predictive'
                if j == 2:
                    version = 'graph-predictive'
                print("Success rate for [{}] [{}] [{}]: {}".format(version, phase, data_folder, 
                    (total_success_rates[j] / num_evaluation) * 100))
                
                html(num_evaluation, phase, version, data_folder)