import os
import math
import attr
import librosa
import numpy as np

from gym import spaces
from typing import Any, Optional, List, Union
from skimage.measure import block_reduce

from habitat.config import Config
from habitat.core.dataset import Episode, Dataset

from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
    AgentState,
)
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar


@registry.register_sensor(name="SoundEventAudioGoalSensor")
class SoundEventAudioGoalSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "audiogoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2, self._sim.config.AUDIO.RIR_SAMPLING_RATE)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_audiogoal_observation()
    

@registry.register_sensor(name="SoundEventSpectrogramSensor")
class SoundEventSpectrogramSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spectrogram"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self._sim.config.AUDIO.TYPE == "binaural":
            spectrogram = self.compute_spectrogram_binaural(np.ones((2, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))
        elif self._sim.config.AUDIO.TYPE == "diff":
            spectrogram = self.compute_diff_spectrogram(np.ones((2, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))
        elif self._sim.config.AUDIO.TYPE == "diff_gd":
            spectrogram = self.compute_diff_gd_spectrogram(np.ones((2, self._sim.config.AUDIO.RIR_SAMPLING_RATE * 5)))
        else:
            raise NotImplementedError(f"Audio type {self._sim.config.AUDIO.TYPE} not supported")
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=spectrogram.shape,
            dtype=np.float32,
        )
    
    @staticmethod
    def compute_spectrogram_binaural(audio_data, sr=16000, hop_len_s=0.02):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft
        
        channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
        spectrogram = np.stack([
            channel1_magnitude, channel2_magnitude], axis=-1
        )

        return spectrogram
    

    @staticmethod
    def compute_diff_spectrogram(audio_data, sr=16000, hop_len_s=0.02):
        def compute_diff(signal):
            ratio = signal[:, :, 0] / signal[:, :, 1] + 1e-8
            angle = np.angle(ratio)
            amp = np.abs(ratio)
            return angle, amp
        
        spectrogram = np.stack([compute_stft(channel) for channel in audio_data], axis=-1).transpose((1, 0, 2))
        diff_angle, diff_amp = compute_diff(spectrogram)
        feat = np.concatenate((np.abs(spectrogram), diff_angle[:, :, np.newaxis], diff_amp[:, :, np.newaxis]), axis=-1)
        feat = block_reduce(feat, block_size=(4, 4, 1), func=np.mean).transpose((1, 0, 2))
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat


    @staticmethod
    def compute_diff_gd_spectrogram(audio_data, sr=16000, hop_len_s=0.02):
        def compute_diff(signal):
            ratio = signal[:, :, 0] / signal[:, :, 1] + 1e-8
            angle = np.angle(ratio)
            amp = np.abs(ratio)
            return angle, amp
        
        spectrogram = np.stack([compute_stft(channel) for channel in audio_data], axis=-1).transpose((1, 0, 2))
        # the dimension of spectrogram is (frames, freq_bins, channels)
        diff_angle, diff_amp = compute_diff(spectrogram)
        feat = np.concatenate((np.abs(spectrogram), diff_angle[:, :, np.newaxis], diff_amp[:, :, np.newaxis]), axis=-1)

        curr_audio_feat = feat[-101:, ...]
        curr_audio_feat = block_reduce(curr_audio_feat, block_size=(4, 4, 1), func=np.mean).transpose((1, 0, 2))
        
        feat = feat[:500, ...]
        feat = block_reduce(feat, block_size=(5, 4, 1), func=np.mean).transpose((1, 0, 2))
        feat = np.concatenate((feat, curr_audio_feat), axis=1)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat
    

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        if self._sim.config.AUDIO.TYPE == "binaural":
            spectrogram = self._sim.get_current_spectrogram_observation(self.compute_spectrogram_binaural)
        elif self._sim.config.AUDIO.TYPE == "diff":
            spectrogram = self._sim.get_current_spectrogram_observation(self.compute_diff_spectrogram)
        elif self._sim.config.AUDIO.TYPE == "diff_gd":
            spectrogram = self._sim.get_current_spectrogram_observation(self.compute_diff_gd_spectrogram)
        else:
            raise NotImplementedError(f"Audio type {self._sim.config.AUDIO.TYPE} not supported")

        return spectrogram
    

def optional_int(value):
    return int(value) if value is not None else None


@attr.s(auto_attribs=True, kw_only=True)
class SoundEventNavEpisode(NavigationEpisode):
    object_category: str
    sound_id: str
    duration: int = attr.ib(converter=int)
    offset: int = attr.ib(converter=int)
    interval_mean: int = attr.ib(converter=int)
    interval_upper_limit: int = attr.ib(converter=int)
    interval_lower_limit: int = attr.ib(converter=int)

    distractor_sound_id: Optional[str] = attr.ib(default=None)
    distractor_duration: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_offset: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_interval_mean: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_interval_upper_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor_interval_lower_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    distractor: Optional[List[NavigationGoal]] = attr.ib(default=None)

    noise_sound_id: Optional[str] = attr.ib(default=None)
    noise_duration: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_offset: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_interval_mean: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_interval_upper_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_interval_lower_limit: Optional[int] = attr.ib(default=None, converter=optional_int)
    noise_positions: Optional[List[List[float]]] = attr.ib(default=None)

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    agent_state: AgentState
    iou: Optional[float]

    
@attr.s(auto_attribs=True, kw_only=True)
class SoundEventGoal(NavigationGoal):
    object_id: str = attr.ib(default=None)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = attr.ib(factory=list)


@registry.register_sensor(name="SoundEventCategory")
class SenCategory(Sensor):
    cls_uuid: str = "category"

    def __init__(self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._category_mapping = {
            "bathtub": 0, 
            "bed": 1, 
            "cabinet": 2, 
            "chair": 3, 
            "chest_of_drawers": 4, 
            "clothes": 5,
            "counter": 6, 
            "cushion": 7, 
            "fireplace": 8, 
            "picture": 9, 
            "plant": 10, 
            "seating": 11, 
            "shower": 12, 
            "sink": 13, 
            "sofa": 14, 
            "stool": 15, 
            "table": 16, 
            "toilet": 17, 
            "towel": 18, 
            "tv_monitor": 19
        }
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR
    
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=1,
            shape=(len(self._category_mapping.keys()),),
            dtype=bool,
        )
    
    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any) -> Any:
        index = self._category_mapping[episode.object_category]
        onehot = np.zeros(len(self._category_mapping))
        onehot[index] = 1
        return onehot


@registry.register_sensor(name="PoseSensorGD")
class PoseSensorGD(Sensor):
    cls_uuid: str = "pose_gd"

    def __init__(
            self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_time = 0
        self._current_episode_id = None
        super().__init__(config=config)

        self.pose_list = []

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.POSITION
    
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Space:
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(5, 4),
            dtype=np.float32,
        )
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_time = 0.0
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        ep_time = self._episode_time
        self._episode_time += 1.0

        pose = np.array(
            [-agent_position_xyz[2], agent_position_xyz[0], agent_heading[0], ep_time],
            dtype=np.float32
        )
        if len(self.pose_list) == 0:
            for _ in range(5):
                self.pose_list.append(np.zeros_like(pose))
        self.pose_list.pop(0)
        self.pose_list.append(pose)
  
        return self.pose_list


def compute_stft(signal):
            hop_length = 160
            win_length = 400
            n_fft = 512
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            return stft