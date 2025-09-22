import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Union

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class Node:
    def __init__(self, position: np.ndarray):
        self.position = position
        self.parent = None
        self.cost = 0.0

class EnhancedRRTStarDQN:
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: np.ndarray,
        obstacles: List[Union[np.ndarray, Tuple, dict]],
        state_size: int = 9,
        action_size: int = 6,
        max_iterations: int = 20000,
        step_size: float = 0.5,
        search_radius: float = 2.0,
        goal_sample_rate: float = 0.1
    ):
        # Initialize start and goal
        self.start = Node(np.array(start))
        self.goal = Node(np.array(goal))
        self.bounds = np.array(bounds)
        self.obstacles = obstacles
        
        # RRT* parameters
        self.nodes = [self.start]
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        
        # DQN parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def check_collision(self, position: np.ndarray) -> bool:
        """Check if position collides with any obstacle"""
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):  # Box obstacle
                if np.all(position >= obs[:3]) and np.all(position <= obs[3:]):
                    return True
            elif isinstance(obs, dict):  # Dynamic obstacle
                obs_pos = np.array(obs['position'])
                obs_radius = obs['radius']
                if np.linalg.norm(position - obs_pos) <= obs_radius:
                    return True
            else:  # Static spherical obstacle
                obs_pos, obs_radius = obs
                if np.linalg.norm(position - np.array(obs_pos)) <= obs_radius:
                    return True
        return False

    def get_random_point(self) -> np.ndarray:
        """Generate random point within environment bounds"""
        if random.random() < self.goal_sample_rate:
            return self.goal.position
        return np.array([
            np.random.uniform(self.bounds[i][0], self.bounds[i][1])
            for i in range(3)
        ])

    def steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Steer from one point toward another within step_size"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return from_point + direction

    def get_nearest_node(self, point: np.ndarray) -> Node:
        """Find nearest node in the tree"""
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def get_neighbors(self, point: np.ndarray) -> List[Node]:
        """Get neighboring nodes within search radius"""
        return [node for node in self.nodes 
                if np.linalg.norm(node.position - point) <= self.search_radius]

    def optimize_path(self, current_state: np.ndarray) -> int:
        """Use DQN to optimize path selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            return self.policy_net(state).max(1)[1].item()

    def update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles"""
        for obs in self.obstacles:
            if isinstance(obs, dict):
                obs['position'] = np.array(obs['position']) + np.array(obs['velocity'])

    def plan_path(self) -> Optional[np.ndarray]:
        """Main RRT* with DQN path planning algorithm"""
        for i in range(self.max_iterations):
            # Update dynamic obstacles
            self.update_dynamic_obstacles()
            
            # Sample random point
            random_point = self.get_random_point()
            nearest_node = self.get_nearest_node(random_point)
            new_position = self.steer(nearest_node.position, random_point)
            
            if self.check_collision(new_position):
                continue
            
            new_node = Node(new_position)
            neighbors = self.get_neighbors(new_position)
            
            # Use DQN for optimization
            state = np.concatenate([new_position, nearest_node.position, self.goal.position])
            action = self.optimize_path(state)
            
            # Choose best parent
            min_cost = float('inf')
            min_parent = None
            
            for neighbor in neighbors:
                cost = neighbor.cost + np.linalg.norm(new_position - neighbor.position)
                if cost < min_cost and not self.check_collision(new_position):
                    min_cost = cost
                    min_parent = neighbor
            
            if min_parent is None:
                continue
                
            new_node.parent = min_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            
            # Rewire
            for neighbor in neighbors:
                cost = new_node.cost + np.linalg.norm(neighbor.position - new_position)
                if cost < neighbor.cost and not self.check_collision(neighbor.position):
                    neighbor.parent = new_node
                    neighbor.cost = cost
            
            # Check goal
            if np.linalg.norm(new_position - self.goal.position) < self.step_size:
                if not self.check_collision(self.goal.position):
                    self.goal.parent = new_node
                    return self.smooth_path(self.extract_path())
            
            # Update DQN
            self.update_dqn()
            
        return None

    def extract_path(self) -> List[np.ndarray]:
        """Extract the path from start to goal"""
        path = []
        current = self.goal
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]

    def smooth_path(self, path: List[np.ndarray]) -> np.ndarray:
        """Enhanced path smoothing with safety constraints and curvature optimization"""
        if path is None or len(path) < 3:
            return path
            
        path = np.array(path)
        
        # Smoothing parameters
        max_iterations = 100
        smoothing_factor = 0.15
        min_clearance = 0.5
        convergence_threshold = 1e-3
        
        smoothed_path = path.copy()
        num_points = len(smoothed_path)
        
        for iteration in range(max_iterations):
            path_modified = False
            
            # Skip endpoints to preserve start and goal
            for i in range(1, num_points - 1):
                original_point = smoothed_path[i].copy()
                
                # Calculate centroid of neighboring points
                neighboring_centroid = (smoothed_path[i-1] + smoothed_path[i+1]) / 2
                
                # Calculate smoothing vector
                smooth_vector = neighboring_centroid - smoothed_path[i]
                
                # Apply smoothing with factor
                new_point = smoothed_path[i] + smoothing_factor * smooth_vector
                
                # Check clearance from obstacles
                is_safe = True
                min_dist = float('inf')
                
                # Check obstacle clearance
                for obs in self.obstacles:
                    if isinstance(obs, np.ndarray):  # Box obstacle
                        mins = obs[:3]
                        maxs = obs[3:]
                        dist = np.min([
                            np.linalg.norm(np.maximum(mins - new_point, 0) +
                                         np.maximum(new_point - maxs, 0))
                        ])
                    else:  # Spherical obstacle
                        if isinstance(obs, dict):
                            center = obs['position']
                            radius = obs['radius']
                        else:
                            center, radius = obs
                        dist = np.linalg.norm(new_point - np.array(center)) - radius
                    
                    min_dist = min(min_dist, dist)
                    if min_dist < min_clearance:
                        is_safe = False
                        break
                
                if is_safe:
                    # Check if path segments are collision-free
                    prev_segment = np.linspace(smoothed_path[i-1], new_point, 5)
                    next_segment = np.linspace(new_point, smoothed_path[i+1], 5)
                    
                    segment_safe = True
                    for point in np.vstack((prev_segment, next_segment)):
                        if self.check_collision(point):
                            segment_safe = False
                            break
                    
                    if segment_safe:
                        # Update point if significant change
                        if np.linalg.norm(new_point - original_point) > convergence_threshold:
                            smoothed_path[i] = new_point
                            path_modified = True
            
            # Check for convergence
            if not path_modified:
                break
        
        # Final safety check and interpolation
        final_path = []
        for i in range(len(smoothed_path) - 1):
            # Add current point
            final_path.append(smoothed_path[i])
            
            # Interpolate between points if gap is large
            gap = np.linalg.norm(smoothed_path[i+1] - smoothed_path[i])
            if gap > self.step_size * 2:
                num_interp = int(gap / self.step_size)
                interp_points = np.linspace(smoothed_path[i], smoothed_path[i+1], num_interp + 2)[1:-1]
                
                # Add interpolated points if safe
                for point in interp_points:
                    if not self.check_collision(point):
                        final_path.append(point)
        
        # Add final point
        final_path.append(smoothed_path[-1])
        
        return np.array(final_path)

    def update_dqn(self):
        """Update DQN networks"""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q = reward_batch + self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if len(self.memory) % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def visualize_3d_path(path: np.ndarray, bounds: np.ndarray, obstacles: List, 
                      start: np.ndarray, goal: np.ndarray):
    """Visualize the path and environment in 3D"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    for obs in obstacles:
        if isinstance(obs, np.ndarray):  # Box obstacle
            x, y, z = obs[:3]
            dx, dy, dz = obs[3:] - obs[:3]
            ax.bar3d(x, y, z, dx, dy, dz, color='gray', alpha=0.5)
        else:  # Spherical obstacle
            if isinstance(obs, dict):
                center = obs['position']
                radius = obs['radius']
            else:
                center, radius = obs
            ax.scatter(center[0], center[1], center[2], c='red', alpha=0.5, s=100)
    
    # Plot path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', label='Path')
    
    # Plot start and goal
    ax.scatter(start[0], start[1], start[2], c='green', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='red', s=100, label='Goal')
    
    # Set bounds
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
