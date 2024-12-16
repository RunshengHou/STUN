import os
import time
import random
import subprocess
import numpy as np

import gym
from gym import spaces
from workloads import face_detect
from tqdm import tqdm


class SchedulerEnv(gym.Env):
    def __init__(self, filter_threshold=0.2):
        super(SchedulerEnv, self).__init__()

        # Scheduler parameters: default values and ranges
        self.param_defaults = {
            "latency_ns": 24_000_000,
            "migration_cost_ns": 500_000,
            "min_granularity_ns": 3_000_000,
            "nr_migrate": 32,
            "sched_rr_timeslice_ms": 100,
            "sched_rt_runtime_us": 950_000,
            "sched_rt_period_us": 1_000_000,
            "sched_cfs_bandwidth_slice_us": 5_000,
            "wakeup_granularity_ns": 4_000_000,
        }
        self.param_ranges = {
            "latency_ns": (100_000, 1_000_000_000),
            "migration_cost_ns": (0, 1_000_000_000),
            "min_granularity_ns": (100_000, 1_000_000_000),
            "nr_migrate": (0, 128),
            "sched_rr_timeslice_ms": (0, 1000),
            "sched_rt_runtime_us": (0, 1_000_000),
            "sched_rt_period_us": (1_000_000, 1_000_000),
            "sched_cfs_bandwidth_slice_us": (1, 1_000_000),
            "wakeup_granularity_ns": (0, 1_000_000_000),
        }

        # Initializing filtered parameters and best parameter records
        self.default_performance = 0
        self.filter_threshold = filter_threshold
        self.filtered_params = self.filter_parameters()
        self.best_params = {}

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.filtered_params) * 2)
        self.observation_space = spaces.Box(
            low=np.array([self.param_ranges[key][0] for key in self.filtered_params], dtype=np.int32),
            high=np.array([self.param_ranges[key][1] for key in self.filtered_params], dtype=np.int32),
            dtype=np.int32
        )


    def print_current_sys_params(self):
        print("==============================================")
        print("Current system params:")

        for param in self.param_defaults.keys():
            if param.startswith("sched_"):
                param_path = f"/proc/sys/kernel/{param}"
            else:
                param_path = f"/sys/kernel/debug/sched/{param}"

            cmd = f"sudo cat {param_path}"
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            print(param + " "*(42-len(param)) + f": {result.stdout.strip()}")


    def set_scheduler_param(self, params):
        for param_name, value in params.items():
            if param_name.startswith("sched_"):
                param_path = f"/proc/sys/kernel/{param_name}"
            else:
                param_path = f"/sys/kernel/debug/sched/{param_name}"

            cmd = f"echo {value} | sudo tee {param_path}"
            subprocess.run(cmd, shell=True, check=True, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)


    def filter_parameters(self):
        """Filter parameters with significant performance impact."""
        significant_params = {}
        self.default_performance = self.evaluate_performance(self.param_defaults)
        threshold = self.filter_threshold

        # for param in tqdm(self.param_defaults.keys(), desc="Filtering Parameters"):
        for param in self.param_defaults.keys():
            print(param)
            self.print_current_sys_params()
            default = self.param_defaults[param]
            low, high = self.param_ranges[param]

            # Test performance with min and max values
            test_params = self.param_defaults.copy()
            test_params[param] = low
            min_performance = self.evaluate_performance(test_params)
            
            self.print_current_sys_params()

            test_params[param] = high
            max_performance = self.evaluate_performance(test_params)
            
            self.print_current_sys_params()

            print("default performance:", self.default_performance)
            print("min performance:", min_performance)
            print("max performance:", max_performance)

            # Check if performance impact exceeds threshold
            if abs(min_performance - self.default_performance) / self.default_performance > threshold or \
               abs(max_performance - self.default_performance) / self.default_performance > threshold:
                significant_params[param] = default

        print(f"Filtered Parameters: {list(significant_params.keys())}")
        return significant_params


    def reset(self):
        """Reset environment and initialize filtered parameters."""
        self.params = {}
        for key in self.filtered_params:
            low, high = self.param_ranges[key]
            if random.random() < 0.5:  # Random initialization
                self.params[key] = random.randint(low, high)
            else:  # Use default or best parameter values
                self.params[key] = self.best_params.get(key, self.param_defaults[key])
        return np.array([self.params[key] for key in self.filtered_params.values()], dtype=np.int32)


    def step(self, action):
        """Execute action and update environment state."""
        param_keys = list(self.filtered_params.keys())
        param_idx = action // 2
        is_increment = action % 2 == 0

        param_name = param_keys[param_idx]
        step_size = (self.param_ranges[param_name][1] - self.param_ranges[param_name][0]) // 10

        # Update parameter value
        if is_increment:
            self.params[param_name] = min(self.params[param_name] + step_size, self.param_ranges[param_name][1])
        else:
            self.params[param_name] = max(self.params[param_name] - step_size, self.param_ranges[param_name][0])

        # Simulate performance evaluation
        performance = self.evaluate_performance(self.params)
        reward = self.calculate_reward(performance)
        done = performance > 1.2  # Task completes if performance exceeds 120%
        return np.array([self.params[key] for key in self.filtered_params], dtype=np.int32), reward, done, {}


    def evaluate_performance(self, params):
        self.set_scheduler_param(params)

        img_paths = os.listdir("./dataset/BioID-FaceDatabase")
        img_paths = [os.path.join("./dataset/BioID-FaceDatabase", name) for name in img_paths]
        img_paths += img_paths

        start = time.time()
        face_detect(img_paths, None)
        end = time.time()

        return len(img_paths) / (end - start)

    def calculate_reward(self, performance):
        """Calculate reward based on performance."""
        if performance > 1.2 * self.default_performance:
            return 200
        elif performance < 0.8 * self.default_performance:
            return -50
        elif performance > self.default_performance:
            return 100
        else:
            return 0

    def render(self, mode="human"):
        """Visualize the current parameters."""
        self.print_current_sys_params()


if __name__ == "__main__":
    STUN_env = SchedulerEnv()
    # STUN_env.render()
    STUN_env.render()
