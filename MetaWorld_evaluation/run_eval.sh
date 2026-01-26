#!/bin/bash

GPU_ID=$1
PORT=$2
CKPT_DIR=$3

if [ -z "$PORT" ]; then
    PORT=9010
fi

if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR="/mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/checkpoints_reflow_offline/checkpoint_epoch_40"
    echo "[INFO] No checkpoint provided, using default: $CKPT_DIR"
else
    echo "[INFO] Using user-provided checkpoint: $CKPT_DIR"
fi

SESSION="evo_$PORT"

echo "=================================================="
echo "Running Evaluation on GPU: $GPU_ID"
echo "Target Tmux Session: $SESSION"
echo "Port: $PORT"
echo "Checkpoint: $CKPT_DIR"
echo "=================================================="

echo "[INFO] Cleaning up port $PORT..."
lsof -ti:$PORT | xargs -r kill -9

tmux has-session -t $SESSION 2>/dev/null
if [ $? == 0 ]; then
    echo "[INFO] Killing previous tmux session '$SESSION'..."
    tmux kill-session -t $SESSION
fi

tmux new-session -d -s $SESSION
tmux split-window -h -t $SESSION

# ================= SERVER (Pane 0.0) =================
tmux send-keys -t $SESSION:0.0 "source ~/.bashrc" C-m
tmux send-keys -t $SESSION:0.0 "conda activate evo" C-m
tmux send-keys -t $SESSION:0.0 "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
tmux send-keys -t $SESSION:0.0 "cu124" C-m
tmux send-keys -t $SESSION:0.0 "cd /mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/scripts" C-m
tmux send-keys -t $SESSION:0.0 "python Evo1_server.py --port $PORT --ckpt_dir \"$CKPT_DIR\"" C-m


# ================= CLIENT (Pane 0.1) =================
tmux send-keys -t $SESSION:0.1 "source ~/.bashrc" C-m
tmux send-keys -t $SESSION:0.1 "conda activate metaworld" C-m
tmux send-keys -t $SESSION:0.1 "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
tmux send-keys -t $SESSION:0.1 "cd /mnt/data_ssd/zhoufang/code/Evo-1/MetaWorld_evaluation" C-m

tmux send-keys -t $SESSION:0.1 "echo '[INFO] Waiting 20s for Server to load model...'" C-m
tmux send-keys -t $SESSION:0.1 "sleep 60" C-m

CLIENT_CMD="python mt50_evo1_client_prompt.py --port $PORT --ckpt_dir \"$CKPT_DIR\"; \
echo '==========================================='; \
echo '[INFO] Evaluation Finished!'; \
echo '[INFO] Session will auto-close in 10 seconds...'; \
echo '==========================================='; \
sleep 10; \
tmux kill-session -t $SESSION"

tmux send-keys -t $SESSION:0.1 "$CLIENT_CMD" C-m

echo "[SUCCESS] Attaching to tmux..."
tmux attach -t $SESSION