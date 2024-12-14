# TailoredKE
Source Code for TailoredKE

% ## Noted: Inside, there are still some errors, waiting to be sorted out.

# How to Run:
python3 -m experiments.evaluate \
    --alg_name=TAILOREDKE \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --num_edits=1 \
    --use_cache
