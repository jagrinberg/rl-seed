import gym
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv


from gym_minigrid.wrappers import *
env = gym.make('MiniGrid-Empty-8x8-v0')
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs = env.reset() # This now produces an RGB tensor only
print(obs)

env_id = 'MiniGrid-Empty-8x8-v0'
video_folder = 'logs/videos/'
video_length = 100

env = DummyVecEnv([lambda: ImgObsWrapper(gym.make(env_id))])

obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent-{}".format(env_id))

env.reset()
for _ in range(video_length + 1):
  action = [env.action_space.sample()]
  obs, _, _, _ = env.step(action)
# Save the video
env.close()