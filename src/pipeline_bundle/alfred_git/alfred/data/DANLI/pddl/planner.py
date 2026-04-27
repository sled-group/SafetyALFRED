import os
import sys
import io
import subprocess
import time
from pprint import pprint

class PDDLPlanner:
    def __init__(self, fd_path, plan_file='sas_plan', alias='ff-astar', timeout=60):
        self.timeout = timeout
        self.search_options = {
            # Optimal
            'dijkstra': '--heuristic "h=blind(transform=adapt_costs(cost_type=NORMAL))" '
                        '--search "astar(h,cost_type=NORMAL,max_time={})"',
            'max-astar': '--heuristic "h=hmax(transform=adapt_costs(cost_type=NORMAL))"'
                         ' --search "astar(h,cost_type=NORMAL,max_time={})"',

            # Suboptimal
            'ff-astar': '--heuristic "h=ff(transform=adapt_costs(cost_type=NORMAL))" '
                        '--search "astar(h,cost_type=NORMAL,max_time={})"',
            'ff-lazy': '--heuristic "h=ff(transform=adapt_costs(cost_type=PLUSONE))" '
                       '--search "lazy_greedy([h],preferred=[h],max_time={})" ',
        }
        self.plan_file = plan_file
        self.fd_path = fd_path #os.path.join(os.getenv('FAST_DOWNWARD_PATH'), "fast-downward.py")
        self.fd_exec_params = self.search_options[alias].format(timeout)
        # self.fd_exec_params = "--alias seq-sat-lama-2011 "

    def plan(self, domain_file, problem_file, debug=False):
        start_time = time.time()
        command = "{} --overall-time-limit {} --plan-file {} --sas-file {} {} {} {} ".format(
            self.fd_path,  self.timeout*3, self.plan_file, self.plan_file+'_temp', domain_file, problem_file, self.fd_exec_params
        )
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=None, close_fds=True)
        output, error = proc.communicate()
        runtime = time.time() - start_time
        if debug:
            print(command)
            if error:
                print(error.decode('utf-8'))
            print(output.decode('utf-8'))
            print('Search runtime:', runtime)
        if "Solution found." not in output.decode('utf-8'):
            return None, runtime
        plan = self.read_sas(self.plan_file)
        return plan, runtime

    def read_sas(self, filename):
      try:
        with open(filename, 'r') as f:
            plan = f.read()
      except:
        return None
      p = plan.split('\n')[:-2]
      retplan = []
      for act in p:
        tup = act.replace(')','').replace('(','').split(' ')
        tup = tuple(tup)
        retplan.append(tup)
      return retplan


if __name__ == '__main__':
    planner = PDDLPlanner(fd_path='/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/fast-downward-24.06.1/fast-downward.py')
    domain_file = "/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/domain.pddl"
    problem_file = "/mnt/external-ssd/rendered_safety_trajs/train/FloorPlan19/fall_trip_hazard/trial_T20190909_152159_542794/traj_data_safety_traj_fall_trip_hazard_Egg_b0ce4484_Cabinet|-00.61|+00.39|-03.51.json/problem.pddl"
    plan, runtime = planner.plan(domain_file, problem_file, True)
    pprint(plan)
    print(plan)