import abc
import logging
import itertools
import torch
import torch.nn as nn

from soundspaces.tasks.nav import PoseSensor, SpectrogramSensor, LocationBelief, CategoryBelief, Category
from ss_baselines.common.utils import CategoricalNet
from ss_baselines.savi.models.smt_cnn import SMTCNN
from sen_baselines.enmus.models.goal_descriptor import GoalDescriptor
from sen_baselines.enmus.models.msmt_state_encoder import MSMTStateEncoder
from sen_baselines.enmus.models.audio_crnn import AudioCRNN, AudioCRNN_GD
from sen.tasks.nav import PoseSensorGD, SenCategory

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, ext_memory_feats

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks
        )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_masks,
    ):
        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations, rnn_hidden_states, prev_actions,
            masks, ext_memory, ext_memory_masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavMSMTPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=128, **kwargs):
        super().__init__(
            AudioNavMSMTNet(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                **kwargs
            ),
            action_space.n
        )


class AudioNavMSMTPolicyWithGD(Policy):
    def __init__(self, observation_space, action_space, hidden_size=128, **kwargs):
        super().__init__(
            AudioNavMSMTNetWithGD(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                **kwargs
            ),
            action_space.n
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavMSMTNet(Net):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,
        use_pretrained=False,
        audio_pretrained_path='',
        visual_pretrained_path='',
        use_belief_as_goal=False,
        use_label_belief=False,
        use_location_belief=False,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input

        assert SpectrogramSensor.cls_uuid in observation_space.spaces
        self.goal_encoder = AudioCRNN(observation_space, 512, 'spectrogram')
        self.downsample = nn.Linear(512, 128)
        audio_feature_dims = 128

        self.visual_encoder = SMTCNN(observation_space)
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims

        if self._use_category_input:
            nfeats += 20

        # Add pose observations to the memory
        assert PoseSensor.cls_uuid in observation_space.spaces
        if PoseSensor.cls_uuid in observation_space.spaces:
            pose_dims = observation_space.spaces[PoseSensor.cls_uuid].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
        else:
            pose_indices = None

        self._feature_size = nfeats

        self.smt_state_encoder = MSMTStateEncoder(
            nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            **kwargs
        )

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(audio_pretrained_path != '' and visual_pretrained_path != '')
            self.pretrained_initialization(audio_pretrained_path, visual_pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size

    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory, ext_memory_masks):
        x = self.get_features(observations, prev_actions)

        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :20] = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1)
                else:
                    belief[:, :20] = observations[CategoryBelief.cls_uuid]

            if self._use_location_belief:
                belief[:, 20:22] = observations[LocationBelief.cls_uuid]

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None

        x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=belief)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        return x_att, rnn_hidden_states, x

    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

        
    def pretrained_initialization(self, audio_path, visual_path):
        print("Loading pretrained models...")
        audio_state_dict = torch.load(audio_path)['state_dict']
        visual_state_dict = torch.load(visual_path)['state_dict']

        self.goal_encoder.load_state_dict(
            {
                k[len('actor_critic.net.audio_encoder.'):]: v for k, v in audio_state_dict.items()
                if 'actor_critic.net.audio_encoder.' in k
            },
        )
        self.visual_encoder.rgb_encoder.load_state_dict(
            {
                k[len('actor_critic.net.visual_encoder.rgb_encoder.'):]: v for k, v in visual_state_dict.items()
                if 'actor_critic.net.visual_encoder.rgb_encoder.' in k
            }
        )
        self.visual_encoder.depth_encoder.load_state_dict(
            {
                k[len('actor_critic.net.visual_encoder.depth_encoder.'):]: v for k, v in visual_state_dict.items()
                if 'actor_critic.net.visual_encoder.depth_encoder.' in k
            }
        )


    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavSMCNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        # if self._use_action_encoding:
        #     params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        self.goal_encoder.eval()
        self.visual_encoder.eval()

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.action_encoder(self._get_one_hot(prev_actions)))
        x.append(self.downsample(self.goal_encoder(observations)))
        if self._use_category_input:
            x.append(observations[SenCategory.cls_uuid])

        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=1)

        return x
    

class AudioNavMSMTNetWithGD(Net):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,
        use_pretrained=False,
        audio_pretrained_path='',
        visual_pretrained_path='',
        seld_pretrained_path='',
        use_belief_as_goal=False,
        use_label_belief=False,
        use_location_belief=False,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        use_goal_descriptor=False,
        num_classes=20,
        gd_encoder_type='CRNN',
        use_downsample=True,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input
        self._use_goal_descriptor = use_goal_descriptor
        self._num_classes = num_classes

        assert SpectrogramSensor.cls_uuid in observation_space.spaces
        self.goal_encoder = AudioCRNN_GD(observation_space, 512, 'spectrogram')
        self.downsample = nn.Linear(512, 128)
        audio_feature_dims = 128

        self.visual_encoder = SMTCNN(observation_space)
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims

        if self._use_category_input:
            nfeats += 20

        assert PoseSensor.cls_uuid in observation_space.spaces
        if PoseSensor.cls_uuid in observation_space.spaces:
            pose_dims = observation_space.spaces[PoseSensor.cls_uuid].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            if self._use_goal_descriptor:
                nfeats += 128
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
        else:
            pose_indices = None

        if self._use_goal_descriptor:
            self.goal_descriptor = GoalDescriptor(
                observation_space,
                256,
                'spectrogram',
                PoseSensorGD.cls_uuid,
                self._num_classes,
                encoder_type=gd_encoder_type,
                downsample=use_downsample,
            )
            self.goal_downsample = nn.Linear(256, 128)

        self._feature_size = nfeats

        self.smt_state_encoder = MSMTStateEncoder(
            nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            **kwargs
        )

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(audio_pretrained_path != '' and visual_pretrained_path != '')
            self.pretrained_initialization(audio_pretrained_path, visual_pretrained_path)
            if self._use_goal_descriptor:
                assert(seld_pretrained_path != '')
                self.goal_descriptor.load_cst_parameters(seld_pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size

    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size
    
    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1
    
    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh
        
    def pretrained_initialization(self, audio_path, visual_path):
        print("Loading pretrained models...")
        audio_state_dict = torch.load(audio_path)['state_dict']
        visual_state_dict = torch.load(visual_path)['state_dict']

        self.goal_encoder.load_state_dict(
            {
                k[len('actor_critic.net.audio_encoder.'):]: v for k, v in audio_state_dict.items()
                if 'actor_critic.net.audio_encoder.' in k
            },
        )
        self.visual_encoder.rgb_encoder.load_state_dict(
            {
                k[len('actor_critic.net.visual_encoder.rgb_encoder.'):]: v for k, v in visual_state_dict.items()
                if 'actor_critic.net.visual_encoder.rgb_encoder.' in k
            }
        )
        self.visual_encoder.depth_encoder.load_state_dict(
            {
                k[len('actor_critic.net.visual_encoder.depth_encoder.'):]: v for k, v in visual_state_dict.items()
                if 'actor_critic.net.visual_encoder.depth_encoder.' in k
            }
        )

    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavSMCNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        # if self._use_action_encoding:
        #     params_to_freeze.append(self.action_encoder.parameters())
        if self._use_goal_descriptor:
            params_to_freeze.append(self.goal_descriptor.cst_former.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        self.goal_encoder.eval()
        self.visual_encoder.eval()
        if self._use_goal_descriptor:
            self.goal_descriptor.cst_former.eval()

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x.append(self.action_encoder(self._get_one_hot(prev_actions)))
        x.append(self.downsample(self.goal_encoder(observations)))
        if self._use_category_input:
            x.append(observations[SenCategory.cls_uuid])
        if self._use_goal_descriptor:
            x.append(self.goal_downsample(self.goal_descriptor(observations)))
        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=1)

        return x

    def forward(
            self,
            observations,
            rnn_hidden_states, 
            prev_actions,
            masks,
            ext_memory,
            ext_memory_masks,
    ):
        x = self.get_features(observations, prev_actions)

        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :20] = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1)
                else:
                    belief[:, :20] = observations[CategoryBelief.cls_uuid]

            if self._use_location_belief:
                belief[:, 20:22] = observations[LocationBelief.cls_uuid]

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None

        x_att = self.smt_state_encoder(x, ext_memory, ext_memory_masks, goal=belief)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        return x_att, rnn_hidden_states, x    