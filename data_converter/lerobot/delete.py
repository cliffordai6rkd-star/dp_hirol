import os 
from pathlib import Path
import shutil
import logging


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Deleter:
    @staticmethod
    def delete_episodes(episode, dir_path):
        temp_trash_dir = dir_path + "_trash"
        os.makedirs(temp_trash_dir, exist_ok=True)
        dirname = f"episode_{str(episode).zfill(4)}"  
        # 绝对路径拼接
        episode_path = os.path.join(dir_path, dirname)
        # 检查是否存在并删除
        if os.path.exists(episode_path):
            # @TODO: move to the temp trash dir
            shutil.move(episode_path, temp_trash_dir)
        else:
            logging.warning(f"Failed")

    # def cor_rate(invalid, directory):
    #     invalid_count = len(invalid)
    #     if not os.path.exists(directory):
    #         print(f"目录不存在: {directory}")
    #         return 0
    #     count = 0
    #     for item in os.listdir(directory):
    #         item_path = os.path.join(directory, item)
    #         if os.path.isdir(item_path):
    #             count += 1
    #     if count == 0:
    #         print(f"目录中没有子目录: {directory}")
    #         return 0
    #     valid = count - invalid_count
    #     cor_rate = valid / count
    #     print(f"总数据量: {count}, 无效数据量: {invalid_count}, 正确率: {cor_rate:.2%}")
