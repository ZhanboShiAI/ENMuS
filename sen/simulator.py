import os
import logging
import pickle
import librosa
import numpy as np
import networkx as nx

from abc import ABC
from math import sqrt
from typing import Any, List, Optional
from collections import defaultdict, namedtuple
from gym import spaces
from scipy.signal import fftconvolve
from scipy.io import wavfile
from spaudiopy.sph import rotate_sh

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor, overwrite_config
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from soundspaces.utils import load_metadata
from soundspaces.mp3d_utils import HouseReader

logging.basicConfig(filename="data/log/sen_log.txt", level=logging.ERROR)


class DummySimulator:
    """
    Dummy simulator for avoiding loading the scene meshes when using cached observations.
    """
    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


@registry.register_simulator()
class SoundEventNavSim(Simulator, ABC):
    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sound = None
        self._offset = None
        self._duration = None
        self._audio_index = None
        self._audio_length = None
        self._audio_interval_mean = None
        self._audio_interval_upper_limit = None
        self._audio_interval_lower_limit = None
        self._audio_interval_determine = None

        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._node2index = None

        self._frame_cache = dict()
        self._audiogoal_cache = dict()
        self._spectrogram_cache = dict()
        self._egomap_cache = defaultdict(dict)
        self._scene_observations = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False
        self._instance2label_mapping = None
        self._house_readers = dict()
        self._use_oracle_planner = True
        self._oracle_actions = list()

        self._eps = 1e-8

        self._audio_buffer = []

        self.points, self.graph = load_metadata(self.metadata_dir)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            self._distractor_position_index = None
            self._current_distractor_sound = None
            self._distractor_offset = None
            self._distractor_duration = None
            self._distractor_audio_index = None
            self._distractor_audio_length = None
            self._distractor_interval_mean = None
            self._distractor_interval_upper_limit = None
            self._distractor_interval_lower_limit = None
            self._distractor_sound_interval_determine = None

        if self.config.AUDIO.HAS_NOISE:
            self._noise_position_index = None
            self._current_noise_sound = None
            self._noise_offset = None
            self._noise_duration = None
            self._noise_audio_index = None
            self._noise_audio_length = None
            self._noise_interval_mean = None
            self._noise_interval_upper_limit = None
            self._noise_interval_lower_limit = None
            self._noise_sound_interval_determine = None

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim = DummySimulator()
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)
        else:
            self._sim = habitat_sim.Simulator(config=self.sim_config)
            self.add_acoustic_config()
            self.material_configured = False

    def add_acoustic_config(self):
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        audio_sensor_spec.uuid = "audio_sensor"
        audio_sensor_spec.enableMaterials = False
        if self.config.AUDIO.TYPE in ["binaural", "diff", "diff_gd"]:
            audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
            audio_sensor_spec.channelLayout.channelCount = 2
        else:
            raise ValueError("Unknown audio type")
        audio_sensor_spec.acousticsConfig.sampleRate = self.config.AUDIO.RIR_SAMPLING_RATE
        audio_sensor_spec.acousticsConfig.threadCount = 1
        audio_sensor_spec.acousticsConfig.indirectRayCount = 500
        audio_sensor_spec.acousticsConfig.temporalCoherence = True
        audio_sensor_spec.acousticsConfig.transmission = True
        self._sim.add_sensor(audio_sensor_spec)

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_position",
                "offset",
                "duration",
                "sound_id",
                "sound_interval_mean",
                "sound_interval_upper_limit",
                "sound_interval_lower_limit",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
                "distractor_sound_id",
                "distractor_position",
                "distractor_offset",
                "distractor_duration",
                "distractor_interval_mean",
                "distractor_interval_upper_limit",
                "distractor_interval_lower_limit",
                "noise_sound_id",
                "noise_duration",
                "noise_offset",
                "noise_interval_mean",
                "noise_interval_upper_limit",
                "noise_interval_lower_limit",
                "noise_positions",
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        if self.config.USE_RENDERED_OBSERVATIONS:
            return self._sim.get_agent_state()
        else:
            return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_agent_state(position, rotation)
        else:
            agent = self._sim.get_agent(agent_id)
            new_state = self.get_agent_state(agent_id)
            new_state.position = position
            new_state.rotation = rotation

            new_state.sensor_states = {}
            agent.set_state(new_state, reset_sensors)
            return True

    @property
    def binaural_rir_dir(self):
        return os.path.join(self.config.AUDIO.BINAURAL_RIR_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def ambisonic_rir_dir(self):
        return os.path.join(self.config.AUDIO.AMBISONIC_RIR_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def distractor_sound_dir(self):
        return self.config.AUDIO.DISTRACTOR_SOUND_DIR
    
    @property
    def noise_sound_dir(self):
        return self.config.AUDIO.NOISE_SOUND_DIR

    @property
    def metadata_dir(self):
        return os.path.join(self.config.AUDIO.METADATA_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def current_scene_name(self):
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]
    
    @property
    def current_distractor_sound(self):
        return self._source_sound_dict[self._current_distractor_sound]
    
    @property
    def current_noise_sound(self):
        return self._source_sound_dict[self._current_noise_sound]

    @property
    def is_silent(self):
        if self._episode_step_count < self._offset or self._episode_step_count > self._duration + self._offset:
            return True
        return False

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)
    
    def _get_random_duration(self, mean, upper, lower):
        sigma = min(np.abs(mean - upper), np.abs(mean - lower)) / 2
        duration = int(np.abs(np.random.normal(mean, sigma)))
        if duration == 0:
            duration = 1
        elif duration > upper:
            duration = upper
        elif duration < lower:
            duration = lower
        return duration

    def _generate_sound_intervals(
            self, offset, duration, audio_length, interval_mean, interval_upper_limit, interval_lower_limit
    ):
        sound_intervals = []
        sound_intervals.extend([0] * offset)

        if interval_mean == -1:
            sound_intervals.extend([1] * duration)
        else:
            while len(sound_intervals) < (duration + offset) and len(sound_intervals) < 500:
                sound_intervals.extend([1] * audio_length)
                interval_len = self._get_random_duration(interval_mean, interval_upper_limit, interval_lower_limit)
                sound_intervals.extend([0] * interval_len)

        sound_intervals = sound_intervals[ :duration + offset]

        if len(sound_intervals) < 500:
            sound_intervals.extend([0] * (500 - len(sound_intervals)))

        sound_intervals.extend([0])
        
        return sound_intervals

    def reconfigure(self, config: Config) -> None:
        self.config = config
        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0
        if self.config.AUDIO.EVERLASTING:
            self._duration = 500
        else:
            assert hasattr(self.config.AGENT_0, 'DURATION')
            self._duration = int(self.config.AGENT_0.DURATION)

        if hasattr(self.config.AGENT_0, 'SOUND_INTERVAL_MEAN'):
            self._audio_interval_mean = int(self.config.AGENT_0.SOUND_INTERVAL_MEAN)
            self._audio_interval_upper_limit = int(self.config.AGENT_0.SOUND_INTERVAL_UPPER_LIMIT)
            self._audio_interval_lower_limit = int(self.config.AGENT_0.SOUND_INTERVAL_LOWER_LIMIT)

        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            if hasattr(self.config.AGENT_0, 'DISTRACTOR_OFFSET'):
                self._distractor_offset = int(self.config.AGENT_0.DISTRACTOR_OFFSET)
            else:
                self._distractor_offset = 0
            if self.config.AUDIO.EVERLASTING:
                self._distractor_duration = 500
            else:
                assert hasattr(self.config.AGENT_0, 'DISTRACTOR_DURATION')
                self._distractor_duration = int(self.config.AGENT_0.DISTRACTOR_DURATION)

            if hasattr(self.config.AGENT_0, 'DISTRACTOR_INTERVAL_MEAN'):
                self._distractor_interval_mean = int(self.config.AGENT_0.DISTRACTOR_INTERVAL_MEAN)
                self._distractor_interval_upper_limit = int(self.config.AGENT_0.DISTRACTOR_INTERVAL_UPPER_LIMIT)
                self._distractor_interval_lower_limit = int(self.config.AGENT_0.DISTRACTOR_INTERVAL_LOWER_LIMIT)

        if self.config.AUDIO.HAS_NOISE:
            if hasattr(self.config.AGENT_0, 'NOISE_OFFSET'):
                self._noise_offset = int(self.config.AGENT_0.NOISE_OFFSET)
            else:
                self._noise_offset = 0
            if self.config.AUDIO.EVERLASTING:
                self._noise_duration = 500
            else:
                assert hasattr(self.config.AGENT_0, 'NOISE_DURATION')
                self._noise_duration = int(self.config.AGENT_0.NOISE_DURATION)

            if hasattr(self.config.AGENT_0, 'NOISE_INTERVAL_MEAN'):
                self._noise_interval_mean = int(self.config.AGENT_0.NOISE_INTERVAL_MEAN)
                self._noise_interval_upper_limit = int(self.config.AGENT_0.NOISE_INTERVAL_UPPER_LIMIT)
                self._noise_interval_lower_limit = int(self.config.AGENT_0.NOISE_INTERVAL_LOWER_LIMIT)
        
        self._audio_index = 0
        is_same_sound = config.AGENT_0.SOUND_ID == self._current_sound
        if not is_same_sound:
            self._current_sound = self.config.AGENT_0.SOUND_ID
            self._load_single_source_sound()
            logging.debug("Switch to sound {} with duration {} seconds".format(self._current_sound, self._duration))

        self._audio_interval_determine = self._generate_sound_intervals(
            self._offset,
            self._duration,
            self._audio_length,
            self._audio_interval_mean,
            self._audio_interval_upper_limit,
            self._audio_interval_lower_limit
        )

        self._audio_buffer = []

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

            if self.config.USE_RENDERED_OBSERVATIONS:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            else:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                if not self.config.USE_RENDERED_OBSERVATIONS:
                    self.add_acoustic_config()
                    self.material_configured = False
                self._update_agents_state()
                self._frame_cache = dict()

            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)
            self._position_to_index_mapping = dict()
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node
            self._instance2label_mapping = None

        if not self.config.USE_RENDERED_OBSERVATIONS:
            audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
            audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.GOAL_POSITION) + np.array([0, 1.5, 0]))
            if not self.material_configured:
                audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")
                self.material_configured = True

        if not is_same_scene or not is_same_sound:
            self._audiogoal_cache = dict()
            self._spectrogram_cache = dict()

        self._episode_step_count = 0
        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.GOAL_POSITION)
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))
        else:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)
        
        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            self._distractor_audio_index = 0
            is_same_distractor_sound = config.AGENT_0.DISTRACTOR_SOUND_ID == self._current_distractor_sound
            if not is_same_distractor_sound:
                self._current_distractor_sound = self.config.AGENT_0.DISTRACTOR_SOUND_ID
                self._load_single_distractor_sound()
                logging.debug("Switch to distractor sound {} with duration {} seconds".
                              format(self._current_distractor_sound, self._distractor_duration))
                
            self._distractor_sound_interval_determine = self._generate_sound_intervals(
                self._distractor_offset,
                self._distractor_duration,
                self._distractor_audio_length,
                self._distractor_interval_mean,
                self._distractor_interval_upper_limit,
                self._distractor_interval_lower_limit
            )
            
            self._distractor_position_index = self._position_to_index(self.config.AGENT_0.DISTRACTOR_POSITION)

        if self.config.AUDIO.HAS_NOISE:
            self._noise_audio_index = 0
            is_same_noise_sound = config.AGENT_0.NOISE_SOUND_ID == self._current_noise_sound
            if not is_same_noise_sound:
                self._current_noise_sound = self.config.AGENT_0.NOISE_SOUND_ID
                self._load_single_noise_sound()
                logging.debug("Switch to noise sound {} with duration {} seconds".
                              format(self._current_noise_sound, self._noise_duration)
                )
            self._noise_sound_interval_determine = self._generate_sound_intervals(
                self._noise_offset,
                self._noise_duration,
                self._noise_audio_length,
                self._noise_interval_mean,
                self._noise_interval_upper_limit,
                self._noise_interval_lower_limit
            )
            noise_positions = self.config.AGENT_0.NOISE_POSITIONS
            noise_position_index = []
            for noise_position in noise_positions:
                noise_position_index.append(self._position_to_index(noise_position))
            self._noise_position_index = noise_position_index
        
        if self._use_oracle_planner:
            self._oracle_actions = self.compute_oracle_actions()

        logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index, self.get_orientation()))

    def compute_semantic_index_mapping(self):
        # obtain mapping from instance id to semantic label id
        if isinstance(self._sim, DummySimulator):
            if self._current_scene not in self._house_readers:
                self._house_readers[self._current_sound] = HouseReader(self._current_scene.replace('.glb', '.house'))
            reader = self._house_readers[self._current_sound]
            instance_id_to_label_id = reader.compute_object_to_category_index_mapping()
        else:
            scene = self._sim.semantic_scene
            instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        self._instance2label_mapping = np.array([instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id))])

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            assert not self.config.USE_RENDERED_OBSERVATIONS
            sim_obs = self._sim.get_sensor_observations()
            for sensor in sim_obs:
                sim_obs[sensor] = sim_obs[sensor]
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        logging.debug('Reset simulation')
        if self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)
        else:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._previous_step_collided = False
        if action == HabitatSimActions.STOP:
            self._is_episode_active = False
        else:
            prev_position_index = self._receiver_position_index
            prev_rotation_angle = self._rotation_angle
            if action == HabitatSimActions.MOVE_FORWARD:
                # the agent initially faces -Z by default
                self._previous_step_collided = True
                for neighbor in self.graph[self._receiver_position_index]:
                    p1 = self.graph.nodes[self._receiver_position_index]['point']
                    p2 = self.graph.nodes[neighbor]['point']
                    direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                    if direction == self.get_orientation():
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
            elif action == HabitatSimActions.TURN_LEFT:
                # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
                self._rotation_angle = (self._rotation_angle + 90) % 360
            elif action == HabitatSimActions.TURN_RIGHT:
                self._rotation_angle = (self._rotation_angle - 90) % 360

            if self.config.CONTINUOUS_VIEW_CHANGE:
                intermediate_observations = list()
                fps = self.config.VIEW_CHANGE_FPS
                if action == HabitatSimActions.MOVE_FORWARD:
                    prev_position = np.array(self.graph.nodes[prev_position_index]['point'])
                    current_position = np.array(self.graph.nodes[self._receiver_position_index]['point'])
                    for i in range(1, fps):
                        intermediate_position = prev_position + i / fps * (current_position - prev_position)
                        self.set_agent_state(intermediate_position.tolist(), quat_from_angle_axis(np.deg2rad(
                                            self._rotation_angle), np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)
                else:
                    for i in range(1, fps):
                        if action == HabitatSimActions.TURN_LEFT:
                            intermediate_rotation = prev_rotation_angle + i / fps * 90
                        elif action == HabitatSimActions.TURN_RIGHT:
                            intermediate_rotation = prev_rotation_angle - i / fps * 90
                        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                             quat_from_angle_axis(np.deg2rad(intermediate_rotation),
                                                                  np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)

            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        self._episode_step_count += 1

        # log debugging info
        logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_index, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)

        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        if self.config.CONTINUOUS_VIEW_CHANGE:
            observations['intermediate'] = intermediate_observations

        return observations

    def get_orientation(self):
        # convert the rotation angle. the orientation is +X forward, 
        # +Z rightward, counter-clockwise
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    @property
    def azimuth_angle(self):
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360
    
    @property
    def azimuth_angle_ambisonic(self):
        # this is the angle used to calculate the ambisonic rir
        # -X forward, -Z rightward, counter-clockwise
        # so +Z is 90, +X is 180, -Z is 270
        return (self._rotation_angle - 90) % 360

    @property
    def reaching_goal(self):
        return self._source_position_index == self._receiver_position_index

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sound = sound_file.split('.')[0]
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound] = audio_data
            self._audio_length = audio_data.shape[0] // self.config.AUDIO.RIR_SAMPLING_RATE

    def _load_single_source_sound(self):
        if self._current_sound not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, self._current_sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE, mono=True)
            self._audio_length = audio_data.shape[0] // sr
            self._source_sound_dict[self._current_sound] = audio_data[: self._audio_length * sr]
        else:
            self._audio_length = self._source_sound_dict[self._current_sound].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE
        assert self._audio_length > 0, "Goal audio length shorter than 1s, not enough for convolve"


    def _load_single_distractor_sound(self):
        if self._current_distractor_sound not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.distractor_sound_dir, self._current_distractor_sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE, mono=True)
            self._distractor_audio_length = audio_data.shape[0] // sr
            self._source_sound_dict[self._current_distractor_sound] = audio_data[: self._distractor_audio_length * sr]
        else:
            self._distractor_audio_length = self._source_sound_dict[self._current_distractor_sound].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE
        assert self._distractor_audio_length > 0, "Distractor audio length shorter than 1s, not enough for convolve"


    def _load_single_noise_sound(self):
        if self._current_noise_sound not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.noise_sound_dir, self._current_noise_sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE, mono=True)
            self._noise_audio_length = audio_data.shape[0] // sr
            self._source_sound_dict[self._current_noise_sound] = audio_data[: self._noise_audio_length * sr]
        else:
            self._noise_audio_length = self._source_sound_dict[self._current_noise_sound].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE
        assert self._noise_audio_length > 0, "Noise audio length shorter than 1s, not enough for convolve"


    def _compute_euclidean_distance_between_sr_locations(self):
        p1 = self.graph.nodes[self._receiver_position_index]['point']
        p2 = self.graph.nodes[self._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        return d

    def _compute_audiogoal(self):
        sr = self.config.AUDIO.RIR_SAMPLING_RATE
        step_idx = self._episode_step_count

        if self.config.AUDIO.TYPE in ["binaural", "diff", "diff_gd"]:
            nb_channel = 2
            cache_rir_path = self.binaural_rir_dir
        else:
            raise ValueError("Unknown audio type")

        if self.config.AUDIO.TYPE == "none":
            return np.zeros((nb_channel, sr))
        
        if (
            step_idx < self._offset or 
            step_idx > self._offset + self._duration
        ):
            audiogoal = np.zeros((nb_channel, sr))
        else:
            if self.config.USE_RENDERED_OBSERVATIONS:
                if self.config.AUDIO.TYPE in ["binaural", "diff", "diff_gd"]:
                    rir_file = os.path.join(cache_rir_path, 
                                                    str(self.azimuth_angle), 
                                                    "{}_{}.wav".format(
                                                        self._receiver_position_index,
                                                        self._source_position_index,
                                                    ))
                else:
                    raise ValueError("Unknown audio type {}".format(self.config.AUDIO.TYPE))

                try:
                    sampling_freq, rir = wavfile.read(rir_file)
                except ValueError:
                    logging.exception("{} file is not readable".format(rir_file))
                    rir = np.zeros((sr, nb_channel))
                if len(rir) == 0:
                    rir = np.zeros((sr, nb_channel))
                
            else:
                audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
                audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.GOAL_POSITION) + np.array([0, 1.5, 0]))
                rir = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))
            
            rir_len = rir.shape[0]
            rir_time_ceil = rir_len // sr + 1

            audio_idx = self._audio_index
            if self._audio_interval_determine[step_idx] == 1:
                self._audio_index = (self._audio_index + 1) % self._audio_length

            if step_idx * sr - rir_len < 0:
                zero_padding = np.zeros(rir_len - step_idx * sr + 1, )
                index = step_idx
            else:
                zero_padding = np.zeros(1, )
                index = rir_time_ceil
                
            goal_sound = np.zeros(1, )
            while index >= 0:
                if self._audio_interval_determine[step_idx] == 1:
                    if audio_idx != -1:
                        step_sound = self.current_source_sound[audio_idx * sr : (audio_idx + 1) * sr]
                    else:
                        step_sound = self.current_source_sound[audio_idx * sr : self._audio_length * sr]
                    audio_idx -= 1
                    if np.abs(audio_idx) >= self._audio_length:
                        audio_idx = 0
                else:
                    step_sound = np.zeros(sr, )

                index -= 1
                step_idx -= 1
                goal_sound = np.concatenate([step_sound, goal_sound])

            goal_sound = np.concatenate([zero_padding, goal_sound])
            goal_sound = goal_sound[-(sr + rir_len) : -1]
            
            audiogoal = np.array([
                fftconvolve(goal_sound, rir[:, channel], mode='valid')
                for channel in range(nb_channel)
            ])
        
        # add distractor sound to audiogoal
        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            if (
                step_idx < self._distractor_offset or 
                step_idx > self._distractor_offset + self._distractor_duration
            ):
                distractor_audiogoal = np.zeros((nb_channel, sr))
            else:
                if self.config.USE_RENDERED_OBSERVATIONS:
                    if self.config.AUDIO.TYPE in ["binaural", "diff", "diff_gd"]:
                        rir_file = os.path.join(cache_rir_path, 
                                                        str(self.azimuth_angle), 
                                                        "{}_{}.wav".format(
                                                            self._receiver_position_index,
                                                            self._distractor_position_index,
                                                        ))
                    else:
                        raise ValueError("Unknown audio type {}".format(self.config.AUDIO.TYPE))

                    try:
                        sampling_freq, rir = wavfile.read(rir_file)
                    except ValueError:
                        logging.exception("{} file is not readable".format(rir_file))
                        rir = np.zeros((sr, nb_channel))
                    if len(rir) == 0:
                        rir = np.zeros((sr, nb_channel))
                    
                else:
                    audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
                    audio_sensor.setAudioSourceTransform(np.array(self.config.AGENT_0.DISTRACTOR_POSITION) + np.array([0, 1.5, 0]))
                    rir = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))
                
                rir_len = rir.shape[0]
                rir_time_ceil = rir_len // sr + 1

                distractor_audio_idx = self._distractor_audio_index
                if self._distractor_sound_interval_determine[step_idx] == 1:
                    self._distractor_audio_index = (self._distractor_audio_index + 1) % self._distractor_audio_length

                if step_idx * sr - rir_len < 0:
                    zero_padding = np.zeros(rir_len - step_idx * sr + 1, )
                    index = step_idx
                else:
                    zero_padding = np.zeros(1, )
                    index = rir_time_ceil

                distractor_sound = np.zeros(1, )
                while index >= 0:
                    if self._distractor_sound_interval_determine[step_idx] == 1:
                        if distractor_audio_idx != -1:
                            step_sound = self.current_distractor_sound[distractor_audio_idx * sr : (distractor_audio_idx + 1) * sr]
                        else:
                            step_sound = self.current_distractor_sound[distractor_audio_idx * sr : self._distractor_audio_length * sr]
                        distractor_audio_idx -= 1
                        if np.abs(distractor_audio_idx) >= self._distractor_audio_length:
                            distractor_audio_idx = 0
                    else:
                        step_sound = np.zeros(sr, )

                    index -= 1
                    step_idx -= 1
                    distractor_sound = np.concatenate([step_sound, distractor_sound])

                distractor_sound = np.concatenate([zero_padding, distractor_sound])
                distractor_sound = distractor_sound[-(sr + rir_len) : -1]
                distractor_audiogoal = np.array([
                    fftconvolve(distractor_sound, rir[:, channel], mode='valid')
                    for channel in range(nb_channel)
                ])

                audiogoal = audiogoal + distractor_audiogoal
        
        if self.config.AUDIO.HAS_NOISE:
            if (
                step_idx < self._noise_offset or 
                step_idx > self._noise_offset + self._noise_duration
            ):
                noise_audiogoal = np.zeros((nb_channel, sr))
            else:
                rir_list = []
                if self.config.USE_RENDERED_OBSERVATIONS:
                    for noise_position_index in self._noise_position_index:
                        if self.config.AUDIO.TYPE in ["binaural", "diff", "diff_gd"]:
                            rir_file = os.path.join(cache_rir_path, 
                                                            str(self.azimuth_angle), 
                                                            "{}_{}.wav".format(
                                                                self._receiver_position_index,
                                                                noise_position_index,
                                                            ))
                        else:
                            raise ValueError("Unknown audio type {}".format(self.config.AUDIO.TYPE))

                        zero_rir = False
                        try:
                            sampling_freq, rir = wavfile.read(rir_file)
                        except ValueError:
                            logging.exception("{} file is not readable".format(rir_file))
                            rir = np.zeros((sr, nb_channel))
                            zero_rir = True
                        except FileNotFoundError:
                            logging.warning("{} file not found".format(rir_file))
                            rir = np.zeros((sr, nb_channel))
                            zero_rir = True
                        if len(rir) == 0:
                            rir = np.zeros((sr, nb_channel))
                            zero_rir = True

                        rir_list.append((rir, zero_rir))   
                else:
                    audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
                    for noise_position_index in self._noise_position_index:
                        audio_sensor.setAudioSourceTransform(np.array(noise_position_index) + np.array([0, 1.5, 0]))
                        rir = np.transpose(np.array(self._sim.get_sensor_observations()["audio_sensor"]))
                        rir_is_zero = np.all(rir == 0)
                        rir_list.append((rir, rir_is_zero))
                
                for rir, is_zero in rir_list:
                    if is_zero:
                        continue
                    
                    rir_len = rir.shape[0]
                    rir_time_ceil = rir_len // sr + 1

                    noise_audio_idx = self._noise_audio_index
                    if self._noise_sound_interval_determine[step_idx] == 1:
                        self._noise_audio_index = (self._noise_audio_index + 1) % self._noise_audio_length

                    if step_idx * sr - rir_len < 0:
                        zero_padding = np.zeros(rir_len - step_idx * sr + 1, )
                        index = step_idx
                    else:
                        zero_padding = np.zeros(1, )
                        index = rir_time_ceil

                    noise_sound = np.zeros(1, )
                    while index >= 0:
                        if self._noise_sound_interval_determine[step_idx] == 1:
                            if noise_audio_idx != -1:
                                step_sound = self.current_noise_sound[noise_audio_idx * sr : (noise_audio_idx + 1) * sr]
                            else:
                                step_sound = self.current_noise_sound[noise_audio_idx * sr : self._noise_audio_length * sr]
                            noise_audio_idx -= 1
                            if np.abs(noise_audio_idx) >= self._noise_audio_length:
                                noise_audio_idx = 0
                        else:
                            step_sound = np.zeros(sr, )

                        index -= 1
                        step_idx -= 1
                        noise_sound = np.concatenate([step_sound, noise_sound])

                    noise_sound = np.concatenate([zero_padding, noise_sound])
                    noise_sound = noise_sound[-(sr + rir_len) : -1]
                    noise_audiogoal = np.array([
                        fftconvolve(noise_sound, rir[:, channel], mode='valid')
                        for channel in range(nb_channel)
                    ])

                    audiogoal = audiogoal + noise_audiogoal
        
        audiogoal = audiogoal + self._eps

        if self.config.AUDIO.TYPE in ["diff_gd"]:
            if len(self._audio_buffer) == 0:
                for _ in range(5):
                    self._audio_buffer.append(np.zeros((nb_channel, sr)))
            self._audio_buffer.append(audiogoal)
            self._audio_buffer.pop(0)
            audiogoal = np.concatenate(self._audio_buffer, axis=1)

        return audiogoal


    def get_egomap_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._egomap_cache[self._current_scene]:
            return self._egomap_cache[self._current_scene][joint_index]
        else:
            return None

    def cache_egomap_observation(self, egomap):
        self._egomap_cache[self._current_scene][(self._receiver_position_index, self._rotation_angle)] = egomap


    def get_current_audiogoal_observation(self):
        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            # by default, does not cache for distractor sound
            audiogoal = self._compute_audiogoal()
        else:
            joint_index = (self._source_position_index, self._receiver_position_index, self.azimuth_angle, self._episode_step_count)
            if joint_index not in self._audiogoal_cache:
                self._audiogoal_cache[joint_index] = self._compute_audiogoal()
            audiogoal = self._audiogoal_cache[joint_index]

        return audiogoal


    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            audiogoal = self.get_current_audiogoal_observation()
            spectrogram = audiogoal2spectrogram(audiogoal)
        else:
            joint_index = (self._source_position_index, self._receiver_position_index, self.azimuth_angle, self._episode_step_count)
            if joint_index not in self._spectrogram_cache:
                audiogoal = self.get_current_audiogoal_observation()
                self._spectrogram_cache[joint_index] = audiogoal2spectrogram(audiogoal)
            spectrogram = self._spectrogram_cache[joint_index]

        return spectrogram


    def geodesic_distance(self, position_a, position_bs, episode=None):
        distances = []
        for position_b in position_bs:
            index_a = self._position_to_index(position_a)
            index_b = self._position_to_index(position_b)
            assert index_a is not None and index_b is not None
            # for the condition when there is no path between view point and goal point
            try:
                path_length = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
            except nx.NetworkXNoPath:
                path_length = np.inf
            distances.append(path_length)

        dist = min(distances)
        if dist != np.inf:
            return dist
        else:
            raise ValueError("Min distance is inf")


    def get_straight_shortest_path_points(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None

        shortest_path = nx.shortest_path(self.graph, source=index_a, target=index_b)
        points = list()
        for node in shortest_path:
            points.append(self.graph.nodes()[node]['point'])
        return points


    def compute_oracle_actions(self):
        start_node = self._receiver_position_index
        end_node = self._source_position_index
        shortest_path = nx.shortest_path(self.graph, source=start_node, target=end_node)
        assert shortest_path[0] == start_node and shortest_path[-1] == end_node
        logging.debug(shortest_path)

        oracle_actions = []
        orientation = self.get_orientation()
        for i in range(len(shortest_path) - 1):
            prev_node = shortest_path[i]
            next_node = shortest_path[i+1]
            p1 = self.graph.nodes[prev_node]['point']
            p2 = self.graph.nodes[next_node]['point']
            direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
            if direction == orientation:
                pass
            elif (direction - orientation) % 360 == 270:
                orientation = (orientation - 90) % 360
                oracle_actions.append(HabitatSimActions.TURN_LEFT)
            elif (direction - orientation) % 360 == 90:
                orientation = (orientation + 90) % 360
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
            elif (direction - orientation) % 360 == 180:
                orientation = (orientation - 180) % 360
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
            oracle_actions.append(HabitatSimActions.MOVE_FORWARD)
        oracle_actions.append(HabitatSimActions.STOP)
        return oracle_actions


    def get_oracle_action(self):
        return self._oracle_actions[self._episode_step_count]


    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def find_nearest_graph_node(self, target_pos):
        from scipy.spatial import cKDTree
        all_points = np.array([self.graph.nodes()[node]['point'] for node in self.graph.nodes()])
        kd_tree = cKDTree(all_points[:, [0, 2]])
        d, ind = kd_tree.query(target_pos[[0, 2]])
        return all_points[ind]

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)
