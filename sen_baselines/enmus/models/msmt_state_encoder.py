import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional

from sen_baselines.enmus.models.msmtransformer import MSMTDecoder, MSMTDecoderLayer

class MSMTStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        pose_indices: Optional[Tuple[int, int]] = None,
        pretraining: bool = False,
        norm_first: bool = True,
        decoder_type: str = "MSMT",
    ):
        super().__init__()
        self._input_size = input_size
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._pose_indices = pose_indices
        self._pretraining = pretraining

        self._pool_ratios = [4, 8, 16, 32]

        if pose_indices is not None:
            pose_dims = pose_indices[1] - pose_indices[0]
            self.pose_encoder = nn.Linear(5, 16)
            input_size += 16 - pose_dims
            self._use_pose_encoding = True
        else:
            self._use_pose_encoding = False

        self.fusion_encoder = nn.Sequential(
            nn.Linear(input_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
        )

        self.encoder_type = None
        self.conv_construct = None

        if decoder_type == "MSMT":
            self.decoder_type = MSMTDecoderLayer
        elif decoder_type == "TRANSFORMER":
            self.decoder_type = nn.TransformerDecoderLayer
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        if decoder_type != "TRANSFORMER":
            self.transformer = nn.Transformer(
                d_model=dim_feedforward,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                custom_decoder=MSMTDecoder(
                    self.decoder_type(
                        d_model=dim_feedforward,
                        num_attention_heads=nhead,
                        feed_forward_expansion_factor=4,
                        conv_expansion_factor=2,
                        feed_forward_dropout_p=dropout,
                        attention_dropout_p=dropout,
                        conv_dropout_p=dropout,
                        conv_kernel_size=31,
                        norm_first=norm_first,
                    ),
                    num_layers=num_decoder_layers,
                    norm=nn.LayerNorm(dim_feedforward),
                ),
            )
        else:
            self.transformer = nn.Transformer(
                d_model=dim_feedforward,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )

    def _convert_masks_to_transformer_format(self, memory_masks):
        return (1 - memory_masks) > 0
    

    def _format_p2t_masks(self, memory_masks):
        masks = [memory_masks]
        for i, ratio in enumerate(self._pool_ratios):
            if i == 0:
                mask = torch.floor(F.avg_pool1d(masks[-1], ratio, ratio))
                masks.append(mask)
            else:
                mask = torch.floor(F.avg_pool1d(masks[-1], 2, 2))
                masks.append(mask)
        masks = torch.cat(masks, dim=1)

        return masks


    def single_forward(self, x, memory, memory_masks, goal=None):
        if self._pretraining:
            memory_masks = torch.cat(
                [torch.zeros_like(memory_masks), torch.ones([memory_masks.shape[0], 1], device=memory_masks.device)],
                dim=1)
        else:
            memory_masks = torch.cat([memory_masks, torch.ones([memory_masks.shape[0], 1], device=memory_masks.device)],
                                     dim=1)

        if self._use_pose_encoding:
            pi, pj = self._pose_indices
            x_pose = x[..., pi:]
            memory_poses = memory[..., pi:]
            x_pose_enc, memory_poses_enc = self._encode_pose(x_pose, memory_poses)
            x = torch.cat([x[..., :pi], x_pose_enc], dim=-1)
            memory = torch.cat([memory[..., :pi], memory_poses_enc], dim=-1)

        memory = torch.cat([memory, x.unsqueeze(0)])
        M, bs = memory.shape[:2]
        memory = self.fusion_encoder(memory.view(M*bs, -1)).view(M, bs, -1)

        t_masks = self._convert_masks_to_transformer_format(memory_masks)
        if "P2T" in self.decoder_type.__name__:
            t_memory_masks = self._format_p2t_masks(memory_masks)
            t_memory_masks = self._convert_masks_to_transformer_format(t_memory_masks)
        else:
            t_memory_masks = t_masks

        if goal is not None:
            x_att = self.transformer(
                memory,
                goal.unsqueeze(0),
                src_key_padding_mask=t_masks,
                memory_key_padding_mask=t_memory_masks,
            )[-1]
        else:
            decode_memory = False
            if decode_memory:
                x_att = self.transformer(
                    memory,
                    memory,
                    src_key_padding_mask=t_masks,
                    tgt_key_padding_mask=t_masks,
                    memory_key_padding_mask=t_memory_masks,
                )[-1]
            else:
                x_att = self.transformer(
                    memory,
                    memory[-1:],
                    src_key_padding_mask=t_masks,
                    memory_key_padding_mask=t_memory_masks,
                    tgt_is_causal=False,
                    memory_is_causal=False,
                )[-1]

        return x_att

    @property
    def hidden_state_size(self):
        return self._dim_feedforward

    def forward(self, x, memory, *args, **kwargs):
        assert x.size(0) == memory.size(1)
        return self.single_forward(x, memory, *args, **kwargs)

    def _encode_pose(self, agent_pose, memory_pose):
        agent_xyh, agent_t = agent_pose[..., :3], agent_pose[..., 3:4]
        memory_xyh, memory_t = memory_pose[..., :3], memory_pose[..., 3:4]

        # Compute relative poses
        agent_rel_xyh = self._compute_relative_pose(agent_xyh, agent_xyh)
        agent_rel_pose = torch.cat([agent_rel_xyh, agent_t], -1)
        memory_rel_xyh = self._compute_relative_pose(agent_xyh.unsqueeze(0), memory_xyh)
        memory_rel_pose = torch.cat([memory_rel_xyh, memory_t], -1)

        # Format pose
        agent_pose_formatted = self._format_pose(agent_rel_pose)
        memory_pose_formatted = self._format_pose(memory_rel_pose)

        # Encode pose
        agent_pose_encoded = self.pose_encoder(agent_pose_formatted)
        M, bs = memory_pose_formatted.shape[:2]
        memory_pose_encoded = self.pose_encoder(
            memory_pose_formatted.view(M * bs, -1)
        ).view(M, bs, -1)

        return agent_pose_encoded, memory_pose_encoded

    def _compute_relative_pose(self, pose_a, pose_b):
        heading_a = -pose_a[..., 2]
        heading_b = -pose_b[..., 2]
        # Compute relative pose
        r_ab = torch.norm(pose_a[..., :2] - pose_b[..., :2], dim=-1)
        phi_ab = torch.atan2(pose_b[..., 1] - pose_a[..., 1], pose_b[..., 0] - pose_a[..., 0])
        phi_ab = phi_ab - heading_a
        x_ab = r_ab * torch.cos(phi_ab)
        y_ab = r_ab * torch.sin(phi_ab)
        heading_ab = heading_b - heading_a
        # Normalize angles to lie between -pi to pi
        heading_ab = torch.atan2(torch.sin(heading_ab), torch.cos(heading_ab))
        # Negate the heading to get angle from x to -y
        heading_ab = -heading_ab

        return torch.stack([x_ab, y_ab, heading_ab], -1) # (..., 3)

    def _format_pose(self, pose):
        """
        Args:
            pose: (..., 4) Tensor containing x, y, heading, time
        """
        x, y, heading, time = torch.unbind(pose, dim=-1)
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        e_time = torch.exp(-time)
        formatted_pose = torch.stack([x, y, cos_heading, sin_heading, e_time], -1)
        return formatted_pose

    @property
    def pose_indices(self):
        return self._pose_indices
