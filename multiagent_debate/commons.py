import time
from math_parsing import parse_math
from prompt import agent_prompt

def query_model(client, agent_context, model_name="gpt-3.5-turbo-0125"):
    try:
        completion = client.chat.completions.create(
                model=model_name,
                messages=agent_context,
                n=1)
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return query_model(agent_context)
    
    content = completion.choices[0].message.content
    return content


def query_hf_model(model, tokenizer, agent_context):
    input_ids = tokenizer.apply_chat_template(
        agent_context,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def parse_question_answer(dataset_name, sample):
    
    if dataset_name == "mmlu":
        question_raw = sample[0]
        a = sample[1]
        b = sample[2]
        c = sample[3]
        d = sample[4]
        answer = sample[5]
        raw_task = tuple(sample.values())
        question = agent_prompt[dataset_name]['question'].format(question_raw, a, b, c, d)
        return question, answer, raw_task
    
    elif dataset_name == "math":
        question_raw = sample["problem"]
        answer = parse_math(sample["solution"])
        question = agent_prompt[dataset_name]['question'].format(question_raw)
        raw_task = sample
        return question, answer, raw_task
    
    elif dataset_name == "chess":
        question_raw = sample["input"]
        last_move = sample['input'].split(' ')[-1]
        question = agent_prompt[dataset_name]['question'].format(question_raw, last_move)
        answer = sample["target"]
        raw_task = sample
        return question, answer, raw_task
    
    elif dataset_name == "mquake":
        question_raw = sample['questions'][0]
        answer = [sample['answer']] + sample['answer_alias']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw,question_raw)
        return question, answer, raw_task
    
    elif dataset_name == "musique":
        question_raw = sample['question']
        answer = [sample['answer']] + sample['answer_aliases']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
        return question, answer, raw_task

    elif dataset_name == "truthfulqa":
        question_raw = sample['question']
        answers_raw = sample['mc1_targets']
        answers = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw)]
        answer = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw) if answers_raw[answer] == 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in answers])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")