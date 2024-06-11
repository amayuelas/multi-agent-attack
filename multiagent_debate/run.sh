python advers.py --dataset truthfulqa \
                 --input_file ./data/truthfulqa/adv_100_3_3_1-gpt-3.5-turbo-gpt-3.5-turbo/adv_truthfulqa_100_3_3_1_0-gpt-3.5-turbo-gpt-3.5-turbo.jsonl \
                 --adv_model 01-ai/Yi-1.5-9B-Chat \
                 --group_model 01-ai/Yi-1.5-9B-Chat \
                 --warning 


python evaluate.py \
       --eval_address results/truthfulqa/adv_warnTrue_100_3_3_1-gpt-3.5-turbo-gpt-3.5-turbo \
       --decision majority