import os


for i in range(0,6):
    os.system('python main.py --algo ppo --use-gae --num-steps 512 --num-processes 2 --env-name "CartPole-v1" --use-linear-lr-decay --eval-interval 50 --log-dir log/cartpole-'+str(i)+' --log-interval 1 --seed '+str(i)+' --num-env-steps 1000000')
    os.system('python main.py --algo ppo --use-gae --num-steps 512 --num-processes 2 --env-name "MountainCar-v0" --use-linear-lr-decay --eval-interval 50 --log-dir log/mountaincar-'+str(i)+' --log-interval 1 --seed '+str(i)+' --num-env-steps 1000000')
    os.system('python main.py --algo ppo --use-gae --num-steps 512 --num-processes 2 --env-name "LunarLander-v2" --use-linear-lr-decay --eval-interval 50 --log-dir log/lunar-'+str(i)+' --log-interval 1 --seed '+str(i)+' --num-env-steps 1000000')

    #')
    #
   
   
    # os.system('python mainseed.py --algo ppo --use-gae --gail --gail-experts-dir gail_experts --num-steps 2048 --num-processes 2 --env-name "InvertedDoublePendulumPyBulletEnv-v0" --use-linear-lr-decay --eval-interval 50 --log-dir log/doubinvpendseed --gail-agent-dir final_models --log-interval 1 --num-env-steps 1000000 --seed '+str(i)+' --demonstration-coef 0.03')
    # os.system('python mainseed.py --algo ppo --use-gae --gail --gail-experts-dir gail_experts --num-steps 2048 --num-processes 2 --env-name "InvertedPendulumPyBulletEnv-v0" --use-linear-lr-decay --eval-interval 50 --log-dir log/invpendswingseed --gail-agent-dir final_models --log-interval 1 --num-env-steps 1000000 --seed '+str(i)+' --demonstration-coef 0.03')

    # os.system('python main.py --algo ppo --use-gae --num-steps 512 --num-processes 2 --env-name "InvertedPendulumSwingupPyBulletEnv-v0" --use-linear-lr-decay --eval-interval 50 --log-dir log/invertswing-'+str(i)+' --log-interval 1 --seed '+str(i)+' --num-env-steps 1000000')
    # os.system('python main.py --algo ppo --use-gae --num-steps 2048 --num-processes 2 --env-name "InvertedDoublePendulumPyBulletEnv-v0" --use-linear-lr-decay --eval-interval 50 --log-dir log/invertdoub-'+str(i)+' --log-interval 1 --seed '+str(i)+' --num-env-steps 1000000')
    # os.system('python main.py --algo ppo --use-gae --num-steps 2048 --num-processes 2 --env-name "InvertedPendulumPyBulletEnv-v0" --use-linear-lr-decay --eval-interval 50 --log-dir log/invertnorm-'+str(i)+' --log-interval 1 --seed '+str(i)+' --num-env-steps 1000000')

