import torch
import numpy as np

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_ppo.ppo.discrete_policy import DiscreteFF

# Import your PPO model definition here
# from UwUBotP2 import YourPolicyClass


from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction

class UwUBotAgent(BaseAgent):
    def initialize_agent(self):
        # --- Update this path to your actual PPO_POLICY.pt checkpoint ---
        checkpoint_path = r"C:/Users/hiori/OneDrive/文档/UwUBot/data/checkpoints/rlgym-ppo-run-1769614790636977300/8101246/PPO_POLICY.pt"
        # DefaultObs for 1v1, adjust if your setup is different
        self.obs_builder = DefaultObs()
        self.action_parser = LookupTableAction()
        # Observation size for DefaultObs 1v1 is usually 135
        obs_size = 135
        n_actions = self.action_parser.get_action_space_size()
        self.policy = DiscreteFF(obs_size, n_actions, [2048, 2048, 1024, 1024])
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.policy.eval()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Convert GameTickPacket to RLGym observation (flat numpy array)
        # This is a minimal conversion using DefaultObs, may need adjustment for your setup
        obs = self.obs_builder.build_obs(packet, 0, None)  # 0 = blue team, None = no previous action
        obs = np.asarray(obs, dtype=np.float32).flatten()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_idx, _ = self.policy.get_action(obs_tensor, deterministic=True)
        # Map action index to controller state
        controller_vals = self.action_parser.parse_actions(np.array([action_idx]))[0]
        controller_state = SimpleControllerState()
        # Map controller_vals to RLBot controls (order: throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
        controller_state.throttle = controller_vals[0]
        controller_state.steer = controller_vals[1]
        controller_state.pitch = controller_vals[2]
        controller_state.yaw = controller_vals[3]
        controller_state.roll = controller_vals[4]
        controller_state.jump = bool(controller_vals[5])
        controller_state.boost = bool(controller_vals[6])
        controller_state.handbrake = bool(controller_vals[7])
        return controller_state

# Note: You must implement your_obs_builder and your_action_parser to match your RLGym setup.
# Place this file in your RLBot Python bot folder and configure RLBot to use UwUBotAgent.
