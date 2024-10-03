import os
import re
import json
from dataloader import get_dataset
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
import argparse
import time
import random
import torch
from pathlib import Path
from evaluate import parse_math
from transformers import AutoTokenizer, AutoModelForCausalLM

from commons import query_model, parse_question_answer, query_hf_model, load_model_tokenizer, query_model_extra
from prompt import agent_prompt, adversary_prompt, optim

def parse_math(text): 

    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1]


def construct_message(dataset_name, agents, question, idx):
    
    prefix_string = agent_prompt[dataset_name]['debate'][0]

    for agent in agents:
        if agent[idx]["role"] == "user": # the conversation has an extra turn because of the system prompt
            assert agent[idx+1]["role"] == "assistant"
            agent_response = agent[idx+1]["content"]
        else:
            agent_response = agent[idx]["content"]

        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + agent_prompt[dataset_name]['debate'][1]
    return {"role": "user", "content": prefix_string}


def construct_adversary_message(dataset_name, agents, answer, argument, context, idx, optim_flag, context_flag):

    prefix_string = agent_prompt[dataset_name]['debate'][0]

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if optim_flag and not context_flag:
        prefix_string = prefix_string + adversary_prompt[dataset_name]['debate'] + " You are also provided with your argument to use. \n" + '\n\n' + "Your Argument: " + argument + '\n\n' + "Your answer: " + f"({answer.upper()})"
    elif not optim_flag and context_flag:
        prefix_string = prefix_string + adversary_prompt[dataset_name]['debate'] + " Use the given answer and the context for your reponse. \n" + "Your answer: " + f"({answer.upper()})" + '\n\n' + "Context: " + context
    elif optim_flag and context_flag:
        prefix_string = prefix_string + adversary_prompt[dataset_name]['debate'] + " Use the given answer, the argument and the context for your reponse. \n" + "Your answer: " + f"({answer.upper()})" +  '\n\n' + "Your Argument: " + argument + '\n\n' + "Context: " + context
    else:
        raise ValueError(f"Invalid mode")
    
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}


def generate_random_chess_move():
    possible_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    possible_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return random.choice(possible_letter) + str(random.choice(possible_number))

def select_adv_answer(dataset_name, answer, raw_task):
    if dataset_name == 'truthfulqa':
        answer_letters = [chr(97 + i) for i in range(len(raw_task['mc1_targets']))]
        answer_letters.remove(answer[0][0])
        incorrect_answers = answer_letters
        adv_answer = random.choice(incorrect_answers)
    elif dataset_name == 'mmlu':
        answer_letters = ['a', 'b', 'c', 'd']
        answer_letters.remove(answer[0][0].lower())
        adv_answer = random.choice(answer_letters)
    elif dataset_name == 'chess':
        random_move = answer[0]
        while random_move in answer:
            random_move = generate_random_chess_move()
        adv_answer = random_move
    elif dataset_name == 'medmcqa':
        answer_letters = ['a', 'b', 'c', 'd']
        answer_letters.remove(answer.lower())
        adv_answer = random.choice(answer_letters)
    elif dataset_name == 'scalr':
        answer_letters = ['a', 'b', 'c', 'd', 'e']
        answer_letters.remove(answer.lower())
        adv_answer = random.choice(answer_letters)        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return adv_answer


def generate_arguments(question, adv_answer, agent_contexts, n_arguments, client, adv_model, idx):

    agent_solutions = ""
    if agent_contexts:
        agent_solutions = "Other agents responses were: \n"
        for agent in agent_contexts:
            if agent[idx]["role"] == "user": # the conversation has an extra turn because of the system prompt
                assert agent[idx+1]["role"] == "assistant"
                agent_response = agent[idx+1]["content"]
            else:
                agent_response = agent[idx]["content"]

            response = "\n\n One agent solution: ```{}```".format(agent_response)
            agent_solutions += response

    context = [
        {"role": "system", "content": optim['argument_generator']['system']},
        {"role": "user", "content": optim['argument_generator']['user'].format(question=question, agent_responses=agent_solutions, adv_answer=adv_answer.upper())}
    ]
        
    completion = query_model_extra(client, context, adv_model, n_repetitions=n_arguments)
    arguments = [choice.message.content for choice in completion.choices]

    return arguments
        


def select_most_persuasive_argument(arguments, question, correct_answer, client, judge_model):

    prompt = optim['argument_selector']['judge']

    dummy_argument = optim['argument_selector']['dummy']

    argument_scores = []
    for argument in arguments:

        context = [
            {"role": "system", "content": prompt.format(
                question=question, 
                debater_1=argument,
                debater_2=dummy_argument.format(answer=correct_answer.upper()))
            }
        ]

        if "mistral" in judge_model or "llama" in judge_model:
            # completion = query_hf_model(adv_model, adv_tokenizer, agent_context)
            raise ValueError(f"Model not supported")
        elif "gpt" in judge_model:
            completion = query_model_extra(client, context, judge_model, logprobs=True, top_logprobs=10, max_tokens=1)
        else:
            raise ValueError(f"Model not supported")

        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        # return top_logprobs

        # Select the most persuasive argument
        prob_1 = -100
        prob_2 = -100
        for top_logprob in top_logprobs:
            if top_logprob.token.lower() == '1' or top_logprob.token.lower() == '(1':
                if top_logprob.logprob > prob_1:
                    prob_1 = top_logprob.logprob
            if top_logprob.token.lower() == '2' or top_logprob.token.lower() == '(2':
                if top_logprob.logprob > prob_2:
                    prob_2 = top_logprob.logprob

        argument_scores.append(prob_1)

    return arguments[argument_scores.index(max(argument_scores))], argument_scores




def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # assert the number of adversaries is <= number of agents
    assert args.n_adversaries <= args.n_agents

    str_group_model = args.group_model
    if '/' in args.group_model: 
        str_group_model = args.group_model.split('/')[-1]
    str_adv_model = args.adv_model
    if '/' in args.adv_model:
        str_adv_model = args.adv_model.split('/')[-1]
    str_judge_model = args.judge_model
    if '/' in args.judge_model:
        str_judge_model = args.judge_model.split('/')[-1]

    str_mode = ""
    if args.run_optim:
        str_mode += "_optim"
    if args.run_context:
        str_mode += "_context"

    # check if out_dir exists, create it if not
    our_dir = Path(args.output_dir, args.dataset, f"TRIAL_adv_plus{str_mode}_{args.n_samples}_{args.n_agents}_{args.n_rounds}_{args.n_adversaries}-{str_group_model}-{str_adv_model}-{str_judge_model}")
    our_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        with open(args.input_file, 'r') as f:
            dataset = [json.loads(line) for line in f]
    else: 
        dataset = get_dataset(dataset_name=args.dataset, n_samples=args.n_samples, context=args.run_context)

    n_agents = args.n_agents
    n_rounds = args.n_rounds

    if "gpt" in args.group_model or "gpt" in args.adv_model:
        client = OpenAI()
    elif "mistral" in args.group_model or "mistral" in args.adv_model or "llama" in args.group_model or "llama" in args.adv_model:
        group_model, group_tokenizer = load_model_tokenizer(args.group_model)
        adv_model, adv_tokenizer = load_model_tokenizer(args.adv_model)
    else:
        raise ValueError(f"Model not supported")
    
    for current_rep in range(args.n_reps):
        print(f"Rep {current_rep}/{args.n_reps}")
        fname = f"adv_plus{str_mode}_{args.dataset}_{args.n_samples}_{args.n_agents}_{args.n_rounds}_{args.n_adversaries}_{current_rep}-{str_group_model}-{str_adv_model}-{str_judge_model}.jsonl"

        with open(our_dir / fname, 'w') as f:
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):

                if args.input_file:
                    sample = sample['raw_task']
                question, answer, raw_task = parse_question_answer(args.dataset, sample)
                question_context = raw_task['context']

                adv_answer = select_adv_answer(args.dataset, answer, raw_task)

                # Initialize the agent contexts
                agent_contexts = []
                for agent in range(n_agents):
                    if agent in range(args.n_adversaries):

                        if args.run_optim:
                            arguments = generate_arguments(question, adv_answer, None, args.n_arguments, client, args.adv_model, None)
                            most_persuasive_argument, argument_scores = select_most_persuasive_argument(arguments, question, adv_answer, client, args.judge_model)

                        if args.run_optim and not args.run_context:
                            adv_prompt = optim['adversary']["init_optim"].format(
                                question=question, 
                                adv_answer=adv_answer.upper(),
                                adv_argument=most_persuasive_argument
                                )
                        elif not args.run_optim and args.run_context:
                            adv_prompt = optim['adversary']["init_context"].format(
                                question=question, 
                                adv_answer=adv_answer.upper(),
                                context=question_context
                                )
                        elif args.run_optim and args.run_context:
                            adv_prompt = optim['adversary']["init_optim_context"].format(
                                question=question, 
                                adv_answer=adv_answer.upper(),
                                adv_argument=most_persuasive_argument,
                                context=question_context
                                )
                        else:
                            raise ValueError(f"Invalid mode")
                        
                        agent_contexts.append(
                            [
                                {"role": "system", "content": adversary_prompt['system']},
                                {"role": "user", "content": adv_prompt}
                                ]
                            )
                    else:
                        agent_contexts.append([{"role": "user", "content": question}])


                for round in range(n_rounds):
                    for agent, agent_context in enumerate(agent_contexts):
                        #TODO: generalize the code. put the if else in a function in commons.py
                        if round != 0:
                            if agent in range(args.n_adversaries):
                                agent_contexts_other = agent_contexts[:agent] + agent_contexts[agent+1:]

                                # Generate arguments
                                most_persuasive_argument = None
                                if args.run_optim:
                                    arguments = generate_arguments(question, adv_answer, agent_contexts_other, args.n_arguments, client, args.adv_model, 2 * round - 1)
                                    most_persuasive_argument, argument_scores = select_most_persuasive_argument(arguments, question, adv_answer, client, args.judge_model)
                                

                                message = construct_adversary_message(args.dataset, agent_contexts_other, adv_answer, most_persuasive_argument, question_context, 2 * round - 1, args.run_optim, args.run_context)
                                
                                agent_context.append(message)
                                if "mistral" in args.adv_model or "llama" in args.adv_model:
                                    completion = query_hf_model(adv_model, adv_tokenizer, agent_context)
                                elif "gpt" in args.adv_model:
                                    completion = query_model(client, agent_context, args.adv_model)
                                else:
                                    raise ValueError(f"Model not supported")

                            else: 
                                agent_contexts_other = agent_contexts[:agent] + agent_contexts[agent+1:]
                                message = construct_message(args.dataset, agent_contexts_other, question, 2 * round - 1)
                                agent_context.append(message)
                                if "mistral" in args.group_model or "llama" in args.group_model:
                                    completion = query_hf_model(group_model, group_tokenizer, agent_context)
                                elif "gpt" in args.group_model:
                                    completion = query_model(client, agent_context, args.group_model)
                                else:
                                    raise ValueError(f"Model not supported")
                                
                            # print(f'round ({round}) - agent [[{agent}], completion: ', completion)

                        else: 
                            if agent in range(args.n_adversaries):
                                if "mistral" in args.adv_model or "llama" in args.adv_model:
                                    completion = query_hf_model(adv_model, adv_tokenizer, agent_context)
                                elif "gpt" in args.adv_model:
                                    completion = query_model(client, agent_context, args.adv_model)
                                else:
                                    raise ValueError(f"Model not supported")
                            else:
                                if "mistral" in args.group_model or "llama" in args.group_model:
                                    completion = query_hf_model(group_model, group_tokenizer, agent_context)
                                elif "gpt" in args.group_model:
                                    completion = query_model(client, agent_context, args.group_model)
                                else:
                                    raise ValueError(f"Model not supported")
                            
                            # print(f'round ({round}) - agent [[{agent}], context: ', agent_context)
                            # print(f'round ({round}) - agent [[{agent}], completion: ', completion)


                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)


                # print("question: ", question)
                # print("answer: ", answer)
                f.write(json.dumps({"id": i, "question": question, "answer": answer, "raw_task": raw_task,  "agent_responses": agent_contexts})+'\n')



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default='truthfulqa', choices=['mmlu', 'chess', 'math', 'mquake', 'musique', 'truthfulqa', 'medmcqa', 'scalr'])
    argparser.add_argument("--input_file", type=str, default='results/truthfulqa/adv_plus_context_100_3_3_1-gpt-4o-gpt-3.5-turbo-gpt-3.5-turbo/adv_plus_context_truthfulqa_100_3_3_1_0-gpt-4o-gpt-3.5-turbo-gpt-3.5-turbo.jsonl', required=False)
    argparser.add_argument("--n_samples", type=int, default=100)
    argparser.add_argument("--n_agents", type=int, default=3)
    argparser.add_argument("--n_rounds", type=int, default=3)
    argparser.add_argument("--n_reps", type=int, default=1)
    argparser.add_argument("--output_dir", type=str, default='results/')
    argparser.add_argument("--n_adversaries", type=int, default=1)
    argparser.add_argument("--group_model", type=str, default='gpt-4o')
    argparser.add_argument("--adv_model", type=str, default='gpt-4o')
    argparser.add_argument("--gpus", type=str, default='0')


    argparser.add_argument("--run_optim", type=bool, default=False)
    argparser.add_argument("--run_context", type=bool, default=False)
    argparser.add_argument("--n_arguments", type=int, default=10)
    argparser.add_argument("--judge_model", type=str, default='gpt-4o')

    args = argparser.parse_args()

    main(args)








