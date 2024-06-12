python advers.py --dataset truthfulqa \
                 --input_file ./data/truthfulqa/adv_100_3_3_1-gpt-3.5-turbo-gpt-3.5-turbo/adv_truthfulqa_100_3_3_1_0-gpt-3.5-turbo-gpt-3.5-turbo.jsonl \
                 --adv_model gpt-3.5-turbo \
                 --group_model gpt-3.5-turbo \
                 --warning 


DECISION_TYPES=(majority judge)
python evaluate.py \
       --eval_address results/truthfulqa/adv_warn_True_100_3_3_1-gpt-4o-gpt-4o \
       --decision majority