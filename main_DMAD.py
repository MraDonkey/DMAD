import json
import random
from model import *
from Dataset import *
from record import *
from PIL import Image
import argparse


eval_prompt = '''Question: {question}. There are the solutions to the question from different agents. 
Solution 1: {solution1}
Solution 2: {solution2}
Solution 3: {solution3}
'''
json_format_prompt = '''Please choose the best solution and output your answer in JSON format, with the format as follows: \{\"Reason\": \"\", \"Index\": \"\"\}. "Index" in the format should only be the index number of the right solution. Please strictly output in JSON format, do not output irrelevant content.'''


def construct_message(agents, question, idx, reasoning):
    
    prefix_string = "These are other answers to the question using different reasoning methods: "

    for agent in agents:
        agent_response = agent[idx]
        response = "\n\n One answer: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    if reasoning == 'io':
        prefix_string = prefix_string + """\n\n Using the answers of different methods as additional information, can you provide your answer to the question? \n {}""".format(question)
    elif reasoning == 'ccot':
        prefix_string += """\n\n Using the answers of different methods as additional information, generate a scene graph in JSON format for the provided image and its associated question.
{}

The scene graph should include:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Obect relationships that are relevant to answering the question.

Just generate the scene graph in JSON format. Do not say extra words.""".format(question)
    elif reasoning == 'ddcot':
        prefix_string += '''Using the answers of different methods as additional information, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. 
{}
        
The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

Sub-answers:
1. <sub-answer 1>
2. <sub-answer 2>
...'''.format(question)
    else:
        raise ValueError(f"{reasoning} is not in ['io', 'ccot', 'ddcot'].")
    return prefix_string


def construct_assistant_message(model, completion):
    if 'gpt' in model:
        content = completion["choices"][0]["message"]["content"]
        return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


ccot_prompt = '''
First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{answer}
'''

ddcot_prompt = '''
First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{answer}
'''


if __name__ == "__main__":
    agents = 3
    rounds = 3
    random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--dataset", type=str, default="ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lecture", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--device", type=str, default = "cuda")
    parser.add_argument("--max_new_tokens", type=int, default = 2048)
    parser.add_argument("--google_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_base_url", type=str, default = "xxx")
    
    args = parser.parse_args()
    args.model_base= None
    args.model_name= None
    args.conv_mode= None
    args.sep= ","
    args.top_p= None
    
    args.system = None
    args.messages = None
    args.query = None
    
    args.model_name, args.tokenizer, args.llava, args.image_processor, args.gemini_model = MLLM_load(args)
    generated_description = []
    
    io_outputs = read_record(args.dataset, args.model, "io", split = args.split, lecture = args.lecture)
    ccot_outputs = read_record(args.dataset, args.model, "ccot", split = args.split, lecture = args.lecture)
    ddcot_outputs = read_record(args.dataset, args.model, "ddcot", split = args.split, lecture = args.lecture)

    for n in range(len(io_outputs)):
        example = io_outputs[n]
        question = example['question']
        if args.dataset == "ScienceQA":
            choices  = example['choices']
            hint = example['hint']
            answer   = example['choices'][example['answer']]
            question = create_prompt(question, choices, context = hint)
            args.image_path = os.path.join(base_path, f"dataset/ScienceQA/{n}.png")
        elif args.dataset == "mm-vet":
            question = create_prompt(question)
            imagename = io_outputs[n]['imagename']
            args.image_path = os.path.join(base_path, 'dataset/mm-vet/images', imagename)
        
        example['scene_graphs'] = ccot_outputs[n]['scene_graphs'][0]
        example['subquestion_answers'] = ddcot_outputs[n]['subquestion_answerses'][0]
        example['io_output'] = io_outputs[n]['outputs'][0]
        example['ccot_output'] = ccot_outputs[n]['outputs'][0]
        example['ddcot_output'] = ddcot_outputs[n]['outputs'][0]
                  
        args.image_file = Image.open(args.image_path)
        
        agent_contexts = [[question] for agent in range(agents)]
        answers_ = [[] for round in range(rounds)]
        
        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                if round == 0:
                    if i == 0:
                        assistant_message = 'Directly answer the question. ' + io_outputs[n][f'outputs'][0]
                        answer = io_outputs[n][f'outputs'][0]
                    elif i == 1:
                        scene_graph = ccot_outputs[n][f'scene_graphs'][0]
                        output = ccot_outputs[n][f'outputs'][0]
                        assistant_message = ccot_prompt.format(scene_graph = scene_graph, answer = output)
                        answer = output
                    elif i == 2:
                        subquestion_answers = ddcot_outputs[n][f'subquestion_answerses'][0]
                        output = ddcot_outputs[n][f'outputs'][0]
                        assistant_message = ddcot_prompt.format(subquestion_answers = subquestion_answers, answer = output)
                        answer = output
                else:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    if i == 0:
                        reasoning = 'io'
                        
                    elif i == 1:
                        reasoning = 'ccot'
                        
                    elif i == 2:
                        reasoning = 'ddcot'
                        
                    message = construct_message(agent_contexts_other, question, 2*round - 1, 'io')
                    
                    reasoning_message = construct_message(agent_contexts_other, question, 2*round - 1, reasoning)
                    args.messages = agent_context + [reasoning_message]
                    assistant_message = MLLM_generate(args)
                    
                    if reasoning == 'io':
                        answer = assistant_message
                        assistant_message = 'Directly answer the question. ' + assistant_message 
                    elif reasoning == 'ccot':
                        scene_graph = assistant_message
                        args.messages = agent_context + [message + f''' The scene graph of the image in JSON format:
{scene_graph}

Use the image and scene graph as context and answer the question.''']
                        answer = MLLM_generate(args)
                        assistant_message = f'''First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{answer}'''
                    elif reasoning == 'ddcot':
                        subquestion_answers = assistant_message
                        if args.dataset == "ScienceQA":
                            args.messages = agent_context + [message + f'''The problem can be deconstructed down to sub-questions.
{subquestion_answers}
According to the sub-questions and sub-answers, give your option of the problem. Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
The answer is A. 
Because ...''']                    
                        elif args.dataset == "mm-vet":
                            args.messages = agent_context + [message + f'''The problem can be deconstructed down to sub-questions.
{subquestion_answers}
Give your answer of the question according to the sub-questions and sub-answers. Just answer the original question, do not say other extra words.''']  
                        
                        answer = MLLM_generate(args)
                        assistant_message = f'''First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{answer}'''
                    
                    agent_context.append(message)
                    
                agent_context.append(assistant_message)
                answers_[round].append(answer)
            
        if args.dataset == "mm-vet":
            example['eval_output'] = []
            example['eval_reason'] = []
            example['eval_index'] = []
            example['eval_solution'] = []
            example['eval_answer'] = []
            for j in range(0, rounds):
                solutions = [agent_context[2*j+1] for agent_context in agent_contexts]
                args.query = eval_prompt.format(question = question, solution1 = solutions[0], solution2 = solutions[1], solution3 = solutions[2])
                args.query += json_format_prompt
                output = MLLM_generate(args)
                example['eval_output'].append(output)
                try:
                    index = re.search(r"ndex\": \"(.*)\"", output).group(1)
                except:
                    index = random.choice(['1', '2', '3'])
                if index not in ['1', '2', '3']:
                    index = random.choice(['1', '2', '3'])
                try:
                    reason = re.search(r"eason\": \"(.*)\"", output).group(1)
                except:
                    reason = ''
                example['eval_index'].append(index)
                example['eval_reason'].append(reason)
                example['eval_solution'].append(solutions[int(index)-1])
                example['eval_answer'].append(answers_[j][int(index)-1])
                
        example['agent_contexts'] = agent_contexts
        example['answers_'] = answers_
        example['usage'] = get_usage(args.model)
        del example["outputs"]
        generated_description.append(example)
        
        if not os.path.exists(os.path.join(base_path, f'outputs_DMAD')):
            os.mkdir(os.path.join(base_path, f'outputs_DMAD'))
        with open(os.path.join(base_path, f'outputs_DMAD/{args.dataset}_{args.split}_{args.model}_{agents}_{rounds}.json'), 'w') as f:
            json.dump(generated_description, f, indent = 4)
