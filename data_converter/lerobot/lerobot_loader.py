from hirol_dataset.lerobot.reader import ActionType, Action_Type_Mapping_Dict, ObservationType
from dataset.lerobot.data_loader_base import DataLoaderBase
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from tqdm import tqdm
import glog as log
import cv2

class LerobotLoader(DataLoaderBase):
    def __init__(self, config, task_dir, json_file_name = "data.json", action_type = ActionType.JOINT_POSITION,
                 observation_type = ObservationType.JOINT_POSITION_ONLY):
        super().__init__(config, task_dir, json_file_name, action_type, observation_type)
        self._observation_type = observation_type
        self._key_sequences = config.get("key_sequences", None)
        self._push_to_repo = self._config.get("push_to_repo", False)
        self._robot_name = config.get("robot_name", "fr3")
        self._repo_name = config.get("repo_name", "peg_in_hole")
        self._task_list = config.get("task_list", None)
        self._contain_depth = config.get(f'contain_depth', False)
        self._custom_prompt = config.get("custom_prompt", None)
        self._output_root_path = config.get("root_path", "../assets/data")
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self._output_root_path = os.path.join(cur_path, self._output_root_path)
        self._load_fps = config.get("fps", 10)
        self._num_writer_thread = config.get(f'num_writer_thread', 1)
        self._num_writer_process = config.get(f'num_writer_process', 1)
        
        # 一种是只给一个task的文件夹
        if self._task_list is None:
            self._task_list = [self._task_dir]
            self._rotation_transform = [self._rotation_transform]
            self._data_type = [self._data_type]
        else: # 另一种是个所有task的文件夹并给每个task的名字在list里
            if self._rotation_transform is None:
                log.info(f'Take all task lists {self._task_list} as real robot data')
                self._rotation_transform = [None] * len(self._task_list)
            elif all([key in self._rotation_transform for key in ["human", "robot"]]):
                # co-trainig
                assert isinstance(self._data_type, list) and len(self._data_type) == len(self._task_list)
                assert isinstance(self._cam_keys, dict) and all([key in self._cam_keys for key in ["human", "robot"]])
                assert all([key in self._skip_nums_step for key in ["human", "robot"]]) 
                human_transform = self._rotation_transform["human"]; robot_transform = self._rotation_transform["robot"]
                human_skip = self._skip_nums_step["human"]; robot_skip = self._skip_nums_step["robot"]
                self._rotation_transform = []; self._skip_nums_step = []
                for i in range(len(self._data_type)):
                    if "human" in self._data_type[i]:
                        self._rotation_transform.append(human_transform)
                        self._skip_nums_step.append(human_skip)
                    elif "robot" in self._data_type[i]:
                        self._rotation_transform.append(robot_transform)
                        self._skip_nums_step.append(robot_skip)
                    else:
                        raise ValueError(f'Data type {self._data_type[i]} is not supported!!!')
            else:
                # human robot sharing the same transform or only one kind of data
                assert all([key not in self._rotation_transform for key in ["human", "robot"]])
                self._rotation_transform = [self._rotation_transform] * len(self._task_list)
            if not isinstance(self._data_type, list):
                self._data_type = [self._data_type] * len(self._task_list)
            for i, task in enumerate(self._task_list):
                cur_task_dir = os.path.join(self._task_dir, task)
                if not os.path.exists(cur_task_dir):
                    raise ValueError(f'{cur_task_dir} did not exist, please check the value of task dir and task list')
                self._task_list[i] = cur_task_dir
        
        assert len(self._task_list) == len(self._rotation_transform)
        assert len(self._data_type) == len(self._task_list)
        if isinstance(self._skip_nums_step, list):
            assert len(self._skip_nums_step) == len(self._task_list)
        if self._custom_prompt:
            assert len(self._custom_prompt) == len(self._task_list), \
            f'custom prompt dimension mismatch, prompt len: {len(self._custom_prompt)} task len: {len(self._task_list)}'
        
    def get_example_feature_dim(self, example_step):
        # try to confirm the images and state length,
        images = example_step.get("colors", {})
        depths = example_step.get("depths", {})
        self._image_keys = []; self._image_resolutions = []
        for key, image in images.items():
            self._image_keys.append(key)
            self._image_resolutions.append(tuple(image.shape))
        if self._contain_depth:
            for key, image in depths.items():
                self._image_keys.append(key)
                self._image_resolutions.append(tuple(image.shape))
        log.info(f'image keys: {self._image_keys}')
        self._obs_states_dim = 0; self._action_dim = 0
        self._ee_states_dim = 0
        obs_states = example_step.get("observations", {})
        actions = example_step.get("actions", {})
        for key, state in obs_states.items():
            # log.info(f'state: {state}, len {len(state)} for {key}')
            self._obs_states_dim += len(state)
        for key, action in actions.items():
            self._action_dim += len(action)
        
        log.info(f'obs state: {self._obs_states_dim}')
        log.info(f'action dim: {self._action_dim}')
    
    def _parse_episode_data(self, task_id, episode_dir, skip_nums_steps):
        task_dir = self._task_list[task_id]
        cam_keys = None
        if isinstance(self._cam_keys, list):
            cam_keys = self._cam_keys
        elif isinstance(self._cam_keys, dict):
            cam_keys = self._cam_keys["human"] if "human" in self._data_type[task_id] else self._cam_keys["robot"]
        else:
            log.debug(f'Take all cameras as the obs images for all task dir since cam keys is {self._cam_keys}')
        cam_keys_transform = self._cam_keys_human2robot if "human" in self._data_type[task_id] else None
        cur_skip_nums_steps = None
        if isinstance(self._skip_nums_step, list):
            cur_skip_nums_steps = self._skip_nums_step[task_id]
        else: cur_skip_nums_steps = self._skip_nums_step
        if cur_skip_nums_steps is None:
            cur_skip_nums_steps = skip_nums_steps
        episode_data, text_info = self.load_episode(task_dir, episode_dir, 
                    cur_skip_nums_steps, self._rotation_transform[task_id], cam_keys, 
                    cam_keys_transform, self._data_type[task_id])
        return episode_data, text_info
    
    def convert_dataset(self):
        skip_nums_steps = int(30.0 / self._load_fps)
        task_dir = self._task_list[0]
        episode_dir = os.listdir(task_dir)[0]
        log.info(f'task dir: {task_dir}, episode dir: {episode_dir}')
        example_episode, _ = self._parse_episode_data(0, episode_dir, skip_nums_steps)
        self.get_example_feature_dim(example_episode[0])
        feature_dicts = {}
        for i, key in enumerate(self._image_keys):
            feature_dicts[key] = dict(dtype = "image",
                                shape = self._image_resolutions[i],
                    names = ["height", "width", "channel"],)
        feature_dicts["state"] = {"dtype": "float64",
                                  "shape": (self._obs_states_dim,),
                                  "names": ["state"],}
        feature_dicts["actions"] = {"dtype": "float64",
                                    "shape": (self._action_dim,),
                                    "names": ["actions"],}
        
        save_path = os.path.join(self._output_root_path, self._repo_name)
        log.info(f'save_path: {save_path}')
        self._lerobot_dataset = LeRobotDataset.create(
            root= save_path,
            repo_id=self._repo_name,
            robot_type=self._robot_name,
            fps=self._load_fps,
            features=feature_dicts,
            image_writer_threads=self._num_writer_thread,
            image_writer_processes=self._num_writer_process,
        )
        
        state_dismatch_list = []; action_dismatch_list = []
        state_step_list = []; action_step_list = []; 
        # 每一个task下的所有episodes
        for task_id, task_dir in tqdm(enumerate(self._task_list), desc=f"processing tasks", unit="task"):
            dirs = os.listdir(task_dir)
            # 同一个task下的每一个episode
            for cur_episode_dir in tqdm(dirs, desc=f"processing episodes", unit="episode"):
                episode_data, text_info = self._parse_episode_data(task_id, cur_episode_dir, skip_nums_steps)
                if episode_data is None or len(episode_data) == 0:
                    continue
                state_wrong_nums = 0; action_wrong_nums = 0 
                for num_step, step in tqdm(enumerate(episode_data), desc=f"processing steps", unit="step"):
                    frame_feature = {}
                    # vision images
                    for image_key in self._image_keys:
                        step_key = "colors" if "color" in image_key else "depths"
                        frame_feature[image_key] = step[step_key][image_key]
                    # obs states
                    frame_feature["state"] = np.array([])
                    obs_states = step["observations"]
                    low_dim_keys = self._key_sequences if self._key_sequences is not None else list(obs_states.keys())
                    for key in low_dim_keys:
                        assert key in obs_states, f"low dim key {key} not in action states {list(obs_states.keys())}"
                        frame_feature["state"] = np.hstack((frame_feature["state"], obs_states[key]))
                    # print(f'state: {frame_feature["state"]}')  # 注释掉频繁的调试输出
                    if len(frame_feature["state"]) != self._obs_states_dim:
                        log.warn(f'{task_dir} {episode_dir} has wrong state dim: {len(frame_feature["state"])} in {num_step}th step')
                        state_wrong = f'{task_dir}_{cur_episode_dir}'
                        if not state_wrong in state_dismatch_list:
                            state_dismatch_list.append(state_wrong)
                        state_wrong_nums += 1
                        continue
                    # actions
                    frame_feature["actions"] = np.array([])
                    action_states = step["actions"]
                    for key in low_dim_keys:
                        assert key in action_states, f"low dim key {key} not in action states {list(action_states.keys())}"
                        frame_feature["actions"] = np.hstack((frame_feature["actions"], action_states[key]))
                    # print(f'actions: {frame_feature["actions"]}')  # 注释掉频繁的调试输出
                    action_dim = len(frame_feature["actions"])
                    if action_dim != self._action_dim:
                        log.warn(f'{task_dir} {episode_dir} has wrong action dim: {action_dim} in {num_step}th step')
                        action_wrong = f'{task_dir}_{cur_episode_dir}'
                        if not action_wrong in state_dismatch_list:
                            action_dismatch_list.append(action_wrong)
                        action_wrong_nums += 1
                        continue
                    text_info = self._custom_prompt[task_id] if self._custom_prompt else text_info
                    frame_feature["task"] = text_info
                    # print(f'frame_feature: {frame_feature}')
                    # raise ValueError
                    
                    self._lerobot_dataset.add_frame(frame=frame_feature)
                if state_wrong_nums != 0:
                    state_step_list.append(state_wrong_nums)
                if action_wrong_nums != 0:
                    action_step_list.append(action_wrong_nums)
                self._lerobot_dataset.save_episode()
                del episode_data
                
                log.info(f'Successfully processed {task_dir}/{cur_episode_dir}'
                         f'and saved to {save_path}')
                log.info(f'{len(self._lack_data_json_list)} lacks the data.json files: {self._lack_data_json_list}')
                log.info(f'state: {state_dismatch_list}, len: {state_step_list}')
                log.info(f'action: {action_dismatch_list}, len: {action_step_list}')
        cv2.destroyAllWindows()
            
        log.info(f'{len(self._lack_data_json_list)} lacks the data.json files: {self._lack_data_json_list}')
        log.info(f'state: {state_dismatch_list}, len: {state_step_list}')
        log.info(f'action: {action_dismatch_list}, len: {action_step_list}')
        
        # Optionally push to the Hugging Face Hub
        if self._push_to_repo:
            self._lerobot_dataset.push_to_hub(
                tags=[self._robot_name, self._repo_name],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )
        return self._lerobot_dataset
    
if __name__ == '__main__':
    import yaml
    task_list = [{'name': "insert_tube", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/lerobot_insert_tube_config.yaml'},
                 {'name': "insert_tube_jps2pose_euler", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/lerobot_insert_tube_jps2pose_euler_config.yaml'},
                 {'name': "insert_tube_jps2delPose_euler", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/lerobot_insert_tube_jps2delPose_euler_config.yaml'},
                 {'name': "insert_tube_pose2delPose_euler", 'task_dir': '/boot/common_data/2025/fr3/1013_inserttube_fr3_pika_50ep', 
                  'cfg_file': 'config/lerobot_insert_tube_pose2delPose_euler_config.yaml'},
                 {'name': "pick_n_place", 'task_dir': '/boot/common_data/2025/fr3/zyx_pick_n_place_500eps', 
                  'cfg_file': 'config/lerobot_pick_n_place_config.yaml'},
                 {'name': "block_stacking", 'task_dir': '/boot/common_data/2025/fr3/1013_stack_blocks_fr3_pika_3dmouse_50ep', 
                  'cfg_file': 'config/lerobot_block_stacking_config.yaml'},
                ]
    
    num_task = len(task_list)
    for i, task in enumerate(task_list):
        print(f'{"=="*5} {i}th task name: {task["name"]}, task dir: {task["task_dir"]}, cfg: {task["cfg_file"]} {"=="*5}')
    get_task = False
    key = -1
    while not get_task:
        key = input(f'Please select the task from 0 - {num_task - 1}: ')
        key = int(key)
        if key >= 0 and key < num_task:
            get_task = True
    print(f'Selected {task_list[key]["name"]} to convert data!!!!')
    
    config_file = task_list[key]["cfg_file"]
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, config_file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'config: {config}')
    task_dir = task_list[key]["task_dir"]
    action_type = config["action_type"]
    action_type = Action_Type_Mapping_Dict[action_type]
    obs_type = config["obs_type"]
    obs_type = ObservationType(obs_type)
   
    lerobot_dataset = LerobotLoader(config, task_dir, action_type=action_type,
                                    observation_type=obs_type)
    dataset = lerobot_dataset.convert_dataset()
    print(f'len dataset: {len(dataset)}')
    print(f'dataset: {dataset}')
    # lerobot自动转换为tensor
    data_sample = next(iter(dataset))
    print(f'data: {data_sample}')
    
    # config.get("root_path", "../assets/data")
    # import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    # root_path = config["root_path"]
    # repo_name = config["repo_name"]
    # root_path = os.path.join(root_path, repo_name)
    # dataset = lerobot_dataset.LeRobotDataset(repo_id=repo_name, root=root_path)
    # print(f'len dataset: {len(dataset)}')
    # print(f'dataset: {dataset}')
    # data_sample = next(iter(dataset))
    # print(f'data: {data_sample}')
    