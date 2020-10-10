import os


for i in range(0,6):
   
    os.system('python mainseed.py --algo ppo --use-gae --gail --gail-experts-dir gail_experts --num-steps 512 --num-processes 2 --use-linear-lr-decay --env-name "InvertedPendulumSwingupPyBulletEnv-v0" --eval-interval 50 --log-dir log/invswingpartseed'+str(i/10)+' --gail-agent-dir final_models --log-interval 1 --num-env-steps 1000000 --demonstration-coef '+str(i/10)+' --seed 1')

