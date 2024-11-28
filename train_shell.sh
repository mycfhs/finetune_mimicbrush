# kill wandb
# ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9

conda activate yyc_mimicbush
cd /home/zl/yyc_workspace/paper_project/MimicBrush


export INSTANCE_DIR="train_data/can/image"
export IP_IMAGE_PATH="./train_data/can/image/00.jpg"

acc0  --main_process_port 12743 ft_lora.py \
  --output_dir="ft_lora/oo_can" \
  --ref_image_path=$IP_IMAGE_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --mixed_precision="no" \
  --instance_prompt="no use"  \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --checkpointing_steps=500 \
  --resume_from_checkpoint="latest" \
  --seed="0" 
  # --validation_prompt="A photo of a dog in a bucket" \
  # --validation_epochs=25 \


  
  