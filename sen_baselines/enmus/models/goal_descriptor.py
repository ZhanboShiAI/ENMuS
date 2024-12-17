import torch
import torch.nn as nn
import torch.nn.functional as F

from sen_baselines.enmus.pretraining.crnn_model import CRNN
from sen_baselines.enmus.pretraining.parameters import get_params

class GoalDescriptor(nn.Module):
    def __init__(
            self, 
            observation_space,
            output_size,
            audiogoal_sensor,
            pose_sensor,
            num_classes,
            encoder_type = 'CRNN',
            downsample=True,
    ):
        super().__init__()
        self._audiogoal_sensor = audiogoal_sensor
        self._pose_sensor = pose_sensor
        self.params = get_params(downsample=downsample)
        self.num_classes = num_classes

        freq_dim = observation_space.spaces[audiogoal_sensor].shape[0]
        seq_len = observation_space.spaces[audiogoal_sensor].shape[1] - 25
        channel = observation_space.spaces[audiogoal_sensor].shape[2]

        if encoder_type == 'CRNN':
            self.cst_former = CRNN(
                in_feat_shape=(
                    seq_len, channel, freq_dim
                ),
                out_dim=num_classes * 3,
                params=self.params,
            )
        else:
            raise ValueError(f"Goal Descriptor unknown encoder type: {encoder_type}")

        self.lstm = nn.LSTM(
            input_size=num_classes * 3,
            hidden_size=output_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, observations):
        audio_observations = observations[self._audiogoal_sensor]
        audio_observations = audio_observations.permute(0, 3, 2, 1)
        audio_observations = audio_observations[:, :, :100, :]

        predict = self.cst_former(audio_observations)
        predict = F.avg_pool1d(predict.permute(0, 2, 1), kernel_size=10).permute(0, 2, 1)

        pos_x, pos_y, pos_z = predict[..., :20], predict[..., 20:40], predict[..., 40:60]
        sed = (torch.sqrt(pos_x**2 + pos_y**2 + pos_z**2) > 0.5).float()
        radians = torch.atan2(pos_y, pos_x)
  
        pos = observations[self._pose_sensor]
        headings = pos[:, :, 2]
        headings = headings.unsqueeze(-1).repeat(1, 1, 20)
        radians = radians + sed * headings

        pos_x = torch.cos(radians)
        pos_y = torch.sin(radians)

        goal_descriptor = torch.cat([sed, pos_x, pos_y], dim=-1)

        _, (hidden_state, _) = self.lstm(goal_descriptor)

        return hidden_state[0]


    def load_cst_parameters(self, model_path):
        print("Load pretrained parameters from: ", model_path)
        self.cst_former.load_state_dict(
            torch.load(
                model_path
            )
        )

        



        