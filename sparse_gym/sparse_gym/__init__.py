from gym.envs.registration import register

register(
    id='SparseMountainCar-v0',
    entry_point='sparse_gym.envs:SparseMountainCarEnv',
)