from typing import Any, Type

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.tasks.nav.nav import NavigationTask
from habitat.core.registry import registry


@registry.register_task(name="SoundEventNav")
class SoundEventNavigationTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)
    


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation

        agent_cfg.GOAL_POSITION = episode.goals[0].position
        agent_cfg.SOUND_ID = episode.sound_id
        agent_cfg.OFFSET = episode.offset
        agent_cfg.DURATION = episode.duration
        agent_cfg.SOUND_INTERVAL_MEAN = episode.interval_mean
        agent_cfg.SOUND_INTERVAL_UPPER_LIMIT = episode.interval_upper_limit
        agent_cfg.SOUND_INTERVAL_LOWER_LIMIT = episode.interval_lower_limit

        if episode.distractor_sound_id is not None:
            agent_cfg.DISTRACTOR_SOUND_ID = episode.distractor_sound_id
            agent_cfg.DISTRACTOR_POSITION = episode.distractor["position"]
            agent_cfg.DISTRACTOR_OFFSET = episode.distractor_offset
            agent_cfg.DISTRACTOR_DURATION = episode.distractor_duration
            agent_cfg.DISTRACTOR_INTERVAL_MEAN = episode.distractor_interval_mean
            agent_cfg.DISTRACTOR_INTERVAL_UPPER_LIMIT = episode.distractor_interval_upper_limit
            agent_cfg.DISTRACTOR_INTERVAL_LOWER_LIMIT = episode.distractor_interval_lower_limit
        
        if episode.noise_sound_id is not None:
            agent_cfg.NOISE_SOUND_ID = episode.noise_sound_id
            agent_cfg.NOISE_DURATION = episode.noise_duration
            agent_cfg.NOISE_OFFSET = episode.noise_offset
            agent_cfg.NOISE_INTERVAL_MEAN = episode.noise_interval_mean
            agent_cfg.NOISE_INTERVAL_UPPER_LIMIT = episode.noise_interval_upper_limit
            agent_cfg.NOISE_INTERVAL_LOWER_LIMIT = episode.noise_interval_lower_limit
            agent_cfg.NOISE_POSITIONS = episode.noise_positions
            
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config
