import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten

def conv_output_dim(dimension, padding, dilation, kernel_size, stride):
    assert len(dimension) == 2, "VISUAL CNN INPUT DIMENSION ERROR"
    output_dimension = []
    for i in range(len(dimension)):
        output_dimension.append(
            int(
                np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                )
            )
        )
    return tuple(output_dimension)

def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

class VisualCNN(nn.Module):
    def __init__(self, observation_space, output_size, extra_rgb) -> None:
        super().__init__()
        if "rgb" in observation_space.spaces:
            self._number_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._number_input_rgb = 0
            
        if "depth" in observation_space.spaces:
            self._number_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._number_input_depth = 0
            
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]
        
        cnn_dimensions = [0, 0]
        if self._number_input_rgb > 0:
            cnn_dimensions = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._number_input_depth > 0:
            cnn_dimensions = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
            
        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dimensions = conv_output_dim(
                    dimension=cnn_dimensions,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32)
                )
            
            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._number_input_depth + self._number_input_rgb,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                Flatten(),
                nn.Linear(64 * cnn_dimensions[0] * cnn_dimensions[1], output_size),
                nn.ReLU(True),
            )
        layer_init(self.cnn)
        
    @property
    def is_blind(self):
        return self._number_input_depth + self._number_input_rgb == 0
    
    def forward(self, observations : dict):
        cnn_input = []
        if self._number_input_rgb > 0:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0
            cnn_input.append(rgb_observations)
            
        if self._number_input_depth > 0:
            depth_observations = observations["depth"]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)
