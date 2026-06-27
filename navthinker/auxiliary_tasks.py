import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default_structured_configs import AuxLossConfig
from habitat_baselines.rl.ppo.policy import Net
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class PeopleCountingLossConfig(AuxLossConfig):
    """People Counting predictive coding loss"""

    max_human_num: int = 6
    loss_scale: float = 0.1

@dataclass
class GuessHumanPositionLossConfig(AuxLossConfig):
    """Guess Human Position predictive coding loss"""

    max_human_num: int = 6
    position_dim: int = 2
    loss_scale: float = 0.1

@dataclass
class FutureTrajectoryPredictionLossConfig(AuxLossConfig):
    """Future Trajectory predictive coding loss"""

    max_human_num: int = 6
    future_step: int = 4
    loss_scale: float = 0.1
    
@baseline_registry.register_auxiliary_loss(name="people_counting")
class PeopleCounting(nn.Module):
    r"""
    People Counting task helps the agent estimate the number of people in the current scene.
    The output is a discrete value between 0 and max_human_num, representing the number of people detected.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        max_human_num: int = 6,
        position_dim: int = 2,
        loss_scale: float = 0.1,
        future_step: int = 4,
    ):
        super().__init__()
        self.max_human_num = max_human_num
        self.loss_scale = loss_scale
        hidden_size = net.output_size
        
        # LSTM to process temporal information
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        # Attention mechanism to focus on important features
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        
        # Classifier to predict the number of people
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, max_human_num + 1),  # Output logits for classes 0 to max_human_num
        )
        
        # CrossEntropy loss for classification
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, aux_loss_state, batch):
        # Use perception embedding as input
        scene_features = aux_loss_state['rnn_output']  # (batch_size, hidden_size)
        
        # Pass through LSTM to capture temporal dependencies
        lstm_output, _ = self.lstm(scene_features)  # (batch_size, hidden_size)

        # Apply Attention mechanism
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  # (batch_size, seq_len, hidden_size)
        
        # Average pooling over the sequence length dimension to aggregate features
        # attn_output_mean = attn_output.mean(dim=1)  # (batch_size, hidden_size)

        # Pass the result through the classifier
        logits = self.classifier(attn_output)  # (batch_size, max_human_num + 1)
        
        logits = torch.clamp(logits, min=-10, max=10)
        # Ground truth is the number of people in the scene
        target = batch["observations"]["human_num_sensor"].squeeze(-1).long()  # (batch_size,)
        
        # Calculate CrossEntropy loss
        ori_loss = self.loss_fn(logits, target) 

        sigmoid_loss = torch.sigmoid(ori_loss)

        loss = self.loss_scale * sigmoid_loss
        
        return dict(loss=loss)

@baseline_registry.register_auxiliary_loss(name="guess_human_position")
class GuessHumanPosition(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        max_human_num: int = 6,
        position_dim: int = 2,
        loss_scale: float = 0.1,
        future_step: int = 4,
    ):
        super().__init__()
        self.loss_scale = loss_scale
        hidden_size = net.output_size
        self.position_dim = position_dim
        self.max_human_num = max_human_num
        
        self.lstm = nn.LSTM(input_size=hidden_size + 1, hidden_size=hidden_size, batch_first=True)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, max_human_num * position_dim),
        )
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, aux_loss_state, batch):

        scene_features = aux_loss_state['rnn_output']  # (t, n, -1)
        human_num_features = batch["observations"]["human_num_sensor"].to(torch.float32)
        features = torch.cat((scene_features, human_num_features), dim=-1)

        lstm_output, _ = self.lstm(features)  # (num_step, hidden_size)
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  # (num_step, hidden_size)

        positions_pred = self.classifier(attn_output)  # (max_human_num * position_dim)
        batch_size = scene_features.size(0)
        positions_pred = positions_pred.view(batch_size, self.max_human_num, self.position_dim)  # (n, max_human_num, position_dim)

        positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]  # (n, num_people, position_dim)
        positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]
        positions_gt_agent0_repeated = positions_gt_agent0.unsqueeze(1).repeat(1, 6, 1)
        positions_gt_relative = positions_gt - positions_gt_agent0_repeated

        mask = (positions_gt != -100.0).all(dim=-1).unsqueeze(-1)  # (n, num_people, 1)
        
        loss_per_position = self.loss_fn(positions_pred, positions_gt_relative)  # (n, max_human_num, position_dim)
        
        masked_loss = loss_per_position * mask  # (batch_size, max_human_num, future_step, position_dim)
        
        # if mask.sum() < 1:
        #     loss = torch.norm(loss_per_position) / 1e5
        # else:
        #     loss = masked_loss.sum() / mask.sum()
        #     max_val = masked_loss.max().detach()
        #     if max_val < 1e-5:
        #         loss = torch.norm(loss_per_position) / 1e5
        #     else:
        #         loss = loss / max_val 
        
        # return dict(loss=loss)

        if mask.sum() < 1:
            loss = torch.norm(loss_per_position) / 1e5
        else:
            loss_mean = masked_loss.mean()
            loss_std = masked_loss.std()

            if loss_std > 1e-5:
                normalized_loss = (masked_loss - loss_mean) / loss_std
            else:
                normalized_loss = masked_loss / loss_mean

            loss = normalized_loss.sum() / mask.sum()
        
        sigmoid_loss = torch.sigmoid(loss)
        
        final_loss = sigmoid_loss * self.loss_scale

        return dict(loss=final_loss)

@baseline_registry.register_auxiliary_loss(name="future_trajectory_prediction")
class FutureTrajectoryPrediction(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        max_human_num: int = 6,
        position_dim: int = 2,
        loss_scale: float = 0.1,
        future_step: int = 4,
    ):
        super().__init__()
        self.max_human_num = max_human_num
        self.position_dim = position_dim
        self.future_step = future_step
        self.loss_scale = loss_scale
        hidden_size = net.output_size

        self.lstm = nn.LSTM(input_size=hidden_size + 1 + max_human_num * position_dim, 
                            hidden_size=hidden_size, 
                            num_layers=2, 
                            bidirectional=True, 
                            batch_first=True)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, max_human_num * future_step * position_dim),
        )

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, aux_loss_state, batch):
        scene_features = aux_loss_state["rnn_output"]  # (batch_size, hidden_size)
        batch_size = scene_features.size(0)
        human_num_features = batch["observations"]["human_num_sensor"].to(torch.float32)  # (batch_size, 1)
        position_features = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :].reshape(batch_size, -1)  # (batch_size, max_human_num * position_dim)
        features = torch.cat((scene_features, human_num_features, position_features), dim=-1)  # (batch_size, hidden_size + 1 + max_human_num * position_dim)
        lstm_output, _ = self.lstm(features)  # (batch_size, 1, hidden_size*2) # .unsqueeze(1)
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  # (1, batch_size, hidden_size*2)

        positions_pred = self.classifier(attn_output)  # (batch_size, max_human_num * future_step * position_dim)
        positions_pred = positions_pred.view(batch_size, self.max_human_num, self.future_step, self.position_dim)  # (batch_size, max_human_num, future_step, position_dim)
        
        positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, -self.future_step:, :]  # (batch_size, num_people, future_step, position_dim)
        positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]  # (batch_size, 2)
        positions_gt_agent0_repeated = positions_gt_agent0.unsqueeze(1).unsqueeze(2).repeat(1, self.max_human_num, self.future_step, 1)  # (batch_size, max_human_num, future_step, 2)
        positions_gt_relative = positions_gt - positions_gt_agent0_repeated  
        
        mask = (positions_gt != -100.0).all(dim=-1).unsqueeze(-1)  # (batch_size, num_people, future_step, 1)
        
        loss_per_position = self.loss_fn(positions_pred, positions_gt_relative)  # (batch_size, max_human_num, future_step, position_dim)
    
        masked_loss = loss_per_position * mask  # (batch_size, max_human_num, future_step, position_dim)
        if mask.sum() < 1:
            loss = torch.norm(loss_per_position) / 1e5
        else:
            loss_mean = masked_loss.mean()
            loss_std = masked_loss.std()

            if loss_std > 1e-5:
                normalized_loss = (masked_loss - loss_mean) / loss_std
            else:
                normalized_loss = masked_loss / loss_mean

            loss = normalized_loss.sum() / mask.sum()
        
        sigmoid_loss = torch.sigmoid(loss)
        
        final_loss = sigmoid_loss * self.loss_scale

        return dict(loss=final_loss)

cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.people_counting",
    group="habitat_baselines/rl/auxiliary_losses",
    name="people_counting",
    node=PeopleCountingLossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.guess_human_position",
    group="habitat_baselines/rl/auxiliary_losses",
    name="guess_human_position",
    node=GuessHumanPositionLossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.future_trajectory_prediction",
    group="habitat_baselines/rl/auxiliary_losses",
    name="future_trajectory_prediction",
    node=FutureTrajectoryPredictionLossConfig,
)