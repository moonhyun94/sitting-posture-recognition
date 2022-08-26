python main_video2.py     \
        -model Conv_5_C3D    \
        -frz True            \
        --gpu_number 1              \
        -epochs 20 \
        -batch_size 1 \
        -lr 1e-5 \
        -pn video \
        -wandb
        
# Conv_5_C3D, 1e-5, dropout 2 adaptive (3,5,5)
# Conv_5_C3D, conv layer freeze, adaptive (3,5,5) ep-20, lr le-5 *

