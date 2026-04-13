from hirol_dataset.lerobot.reader import RerunEpisodeReader

def main():
    task_dir = "/home/yuxuan/Code/hirol/teleoperated_trajectory/fr3/0910/block_stacking"
    episode_number = 10
    
    reader = RerunEpisodeReader(task_dir)
    text = reader.get_episode_text_info(episode_number)
    print(f'text: {text}')
    
    
if __name__ == "__main__":
    main()
    