import abc
from dataset.lerobot.reader import RerunEpisodeReader, ActionType, ObservationType
import glog as log

class DataLoaderBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config, task_dir:str, json_file_name:str="data.json", action_type:ActionType=ActionType.JOINT_POSITION,
                 observation_type=ObservationType.JOINT_POSITION_ONLY):
        self._config = config
        self._action_prediction_step = config.get("action_prediction_step", 2)
        self._skip_nums_step = config.get(f'skip_num_steps', None)
        self._action_type = action_type
        self._obs_type = observation_type
        self._action_ori_type = config.get("action_ori_type", "euler")
        self._rotation_transform = config.get("rotation_transform", None)
        self._contain_ft = config.get(f'contain_ft', False)
        self._cam_keys = config.get("cam_keys", None)
        self._cam_keys_human2robot = config.get("cam_keys_human2robot", None)
        self._state_keys = config.get("state_keys", None)
        self._real_robot_tool_sacle = config.get("real_robot_tool_sacle", 1.0)
        self._data_type = config.get("data_type", "real_robot")
        self._task_dir = task_dir
        self._json_file = json_file_name
        self._lack_data_json_list = []
    
    """
        parse for single episode given task dir and episode dir
    """
    def load_episode(self, task_dir, episode_dir, skip_steps_nums, rotation_transform=None, 
                     camera_keys=None, camera_keys_transform=None, data_type="real_robot"):
        self._episode_reader = RerunEpisodeReader(task_dir=task_dir,
                json_file=self._json_file, action_type=self._action_type,
                action_prediction_step=self._action_prediction_step,
                action_ori_type=self._action_ori_type, observation_type=self._obs_type,
                rotation_transform=rotation_transform, contain_ft=self._contain_ft,
                camera_keys=camera_keys, camera_keys_transformation=camera_keys_transform,
                state_keys=self._state_keys, data_type=data_type, 
                real_robot_tool_sacle=self._real_robot_tool_sacle)
        
        if 'episode' in episode_dir:
            episode_number = int(episode_dir.lstrip("episode_"))
            episode_id = episode_number
            log.info(f'Tring to load the {episode_number}th episode data in {task_dir} for {data_type} with {skip_steps_nums} {camera_keys} {camera_keys_transform}')
            episode_data = self._episode_reader.return_episode_data(episode_number, skip_steps_nums)
            if episode_data is None:
                self._lack_data_json_list.append(f"{task_dir}_{episode_dir}")
                return None, None
        else:
            log.warn(f"{episode_dir} in {task_dir} does not contain episode")
            return None, None
        
        text_info = self._episode_reader.get_episode_text_info(episode_id)
        return episode_data, text_info  
        
    @abc.abstractmethod
    def convert_dataset(self):
        raise NotImplementedError
    