import os
import re
import json
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from commons import query_model, parse_question_answer, query_hf_model
from dataloader import get_dataset
from prompt import agent_prompt


def construct_message(dataset_name, agents, question, idx):
    
    prefix_string = agent_prompt[dataset_name]['debate'][0]

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + agent_prompt[dataset_name]['debate'][1]
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}



def main(args):

    # check if out_dir exists, create it if not
    out_dir = Path(args.output_dir, args.dataset, f"{args.n_samples}_{args.n_agents}_{args.n_rounds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(dataset_name=args.dataset, n_samples=args.n_samples)
    n_agents = args.n_agents
    n_rounds = args.n_rounds

    if "mistral" in args.model_name or "llama" in args.model_name:
        # load tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    elif "gpt" in args.model_name:
        client = OpenAI()
    else:
        raise ValueError(f"Model {args.model_name} not supported")
    
    for current_rep in range(args.n_reps):
        print(f"Rep {current_rep}/{args.n_reps}")
        fname = f"{args.dataset}_{args.n_samples}_{args.n_agents}_{args.n_rounds}_{current_rep}.jsonl"

        with open(out_dir / fname, 'w') as f:
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):

                question, answer, raw_task = parse_question_answer(args.dataset, sample)
                # print('raw_task:', raw_task)

                # Initialize the agent contexts
                agent_contexts = [[{"role": "user", "content": question}] for agent in range(n_agents)]

                for round in range(n_rounds):
                    for agent, agent_context in enumerate(agent_contexts):

                        if round != 0:
                            agent_contexts_other = agent_contexts[:agent] + agent_contexts[agent+1:]
                            message = construct_message(args.dataset, agent_contexts_other, question, 2 * round - 1)
                            agent_context.append(message)

                        if "mistral" in args.model_name or "llama" in args.model_name:
                            completion = query_hf_model(model, tokenizer, agent_context)
                        elif "gpt" in args.model_name:
                            completion = query_model(client, agent_context, model_name=args.model_name)
                        else:
                            raise ValueError(f"Model {args.model_name} not supported")

                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)

                print("question: ", question)
                print("answer: ", answer)
                f.write(json.dumps({"id": i, "question": question, "answer": answer, "raw_task": raw_task,  "agent_responses": agent_contexts})+'\n')



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default='musique', choices=['mmlu', 'chess', 'math', 'mquake', 'musique', 'truthfulqa'])
    argparser.add_argument("--n_samples", type=int, default=100)
    argparser.add_argument("--n_agents", type=int, default=3)
    argparser.add_argument("--n_rounds", type=int, default=3)
    argparser.add_argument("--n_reps", type=int, default=5)
    argparser.add_argument("--output_dir", type=str, default='results/')
    argparser.add_argument("--model_name", type=str, default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4-turbo', 'mistral-7b-instruct-v2', 'llama-3-8b-instruct'])

    args = argparser.parse_args()

    main(args)








