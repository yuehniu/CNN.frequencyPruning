# Train orignal mnist from scratch
python run.py --checkpoint params --epochs 100
# Finetune from pre-trained model
python run.py --admm --restore --checkpoint paramsADMM --restoredir params --epochs 100

# Finetune in frequency domain
python run.py --restore --test

