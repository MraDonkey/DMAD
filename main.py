from model import *
import model
from Dataset import *
from record import record, read_record, base_path
from reasoning import *
import argparse
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.6-13b")
    parser.add_argument("--dataset", type=str, default="ScienceQA")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lecture", type=bool, default=False)
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--device", type=str, default = "cuda")
    parser.add_argument("--max_new_tokens", type=int, default = 2048)
    parser.add_argument("--google_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_api_key", type=str, default = "xxx")
    parser.add_argument("--openai_completions_url", type=str, default = "https://api.openai.com/v1/chat/completions")

    args = parser.parse_args()
    args.model_base= None
    args.model_name= None
    args.conv_mode= None
    args.sep= ","
    args.top_p= None
    
    args.system = None
    args.messages = None
    args.query = None
    
    data = read_dataset(args.dataset, args.split)
    if args.dataset == "ScienceQA":
        data = [d for d in data if d["image"] != None]
    args.model_name, args.tokenizer, args.llava, args.image_processor, args.model_gemini = MLLM_load(args)
    
    logs = read_record(args.dataset, args.model, args.reasoning, split = args.split, lecture = args.lecture)
    begin_num = len(logs)
    num =  begin_num
    if num > 0:
        model.prompt_tokens = logs[-1]["usage"]["prompt_tokens"]
        model.completion_tokens = logs[-1]["usage"]["completion_tokens"]
    if args.dataset == "ScienceQA":
        if args.reasoning == "io":
            for i in range(begin_num, len(data)):
                example = data[i]
                example["num"] = num
                
                question = example["question"]
                choices  = example["choices"]
                
                image_path    = os.path.join(base_path, f"dataset/ScienceQA/{num}.png")
                answer   = example["choices"][example["answer"]]
                if args.lecture:
                    lecture  = example["lecture"]
                    hint     = example["hint"]
                else:
                    lecture = None
                    hint = None

                solution = example["solution"]
                example["image"] = "image"
                
                args.query = create_prompt(question, choices, lecture, hint)
                args.image_path= image_path
                args.image_file = Image.open(image_path)
                
                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output) 
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning = args.reasoning, content = logs, split = args.split, lecture = args.lecture)
        elif args.reasoning == "ccot":
            for i in range(begin_num, len(data)):
                example = data[i]
                example["num"] = num
                
                question = example["question"]
                choices  = example["choices"]

                image_path    = os.path.join(base_path, f"dataset/ScienceQA/{num}.png")
                answer   = example["choices"][example["answer"]]
                if args.lecture:
                    lecture  = example["lecture"]
                    hint     = example["hint"]
                else:
                    lecture = None
                    hint = None
                solution = example["solution"]
                example["image"] = "image"
                
                options = choices
                
                create_scene_graph_prompt = create_ccot_create_scene_graph_prompt(question, choices, lecture, hint)
                args.image_path= image_path
                args.image_file = Image.open(image_path)
                
                outputs = []
                scene_graphs = []
                
                for n in range(0, 3):
                    args.query = create_scene_graph_prompt
                    scene_graph = MLLM_generate(args)
                    args.query = create_ccot_answer_with_scene_graph_prompt(scene_graph, question, options, lecture, hint)
                    output = MLLM_generate(args)
                    scene_graphs.append(scene_graph)
                    outputs.append(output) 
                example["scene_graphs"] = scene_graphs
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)

        elif args.reasoning == "ddcot":    
            for i in range(begin_num, len(data)):
                example = data[i]
                example["num"] = num

                question = example["question"]
                choices  = example["choices"]
                image_path    = os.path.join(base_path, f"dataset/ScienceQA/{num}.png")
                answer   = example["choices"][example["answer"]]
                if args.lecture:
                    lecture  = example["lecture"]
                    hint     = example["hint"]
                else:
                    lecture = None
                    hint = None
                solution = example["solution"]
                example["image"] = "image"
                
                options = choices
                
                create_subquestions_prompt = create_ddcot_create_subquestions_prompt(question, choices, lecture, hint)
                args.image_path= image_path
                args.image_file = Image.open(image_path)
                
                outputs = []
                subquestion_answerses = []

                for n in range(0, 3):
                    args.query = create_subquestions_prompt
                    subquestion_answers = MLLM_generate(args)
                    args.query = create_ddcot_answer_with_subquestions_prompt(subquestion_answers, question, options, lecture, hint)
                    output_ddcot = MLLM_generate(args)
                    subquestion_answerses.append(subquestion_answers)
                    outputs.append(output_ddcot)
                example["subquestion_answerses"] = subquestion_answerses
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)

    elif args.dataset == "mm-vet":
        if args.reasoning == "io":
            for i in range(begin_num, len(data)):
                id = f"v1_{i}"
                example = data[id]
                imagename = data[id]["imagename"]
                img_path = os.path.join(base_path, "dataset/mm-vet/images", imagename)
                question = data[id]["question"]
                print(f"\n{id}")
                print(f"Image: {imagename}")
                
                example["num"] = num

                args.query = create_prompt(question, if_options = False)
                args.image_file= Image.open(img_path)
                args.image_path = img_path

                outputs = []
                for n in range(0, 3):
                    output = MLLM_generate(args)
                    outputs.append(output) 
                example["outputs"] = outputs
                
                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, reasoning = args.reasoning, content = logs, split = args.split, lecture = args.lecture)
        
        elif args.reasoning == "ccot":
            for i in range(begin_num, len(data)):
                id = f"v1_{i}"
                example = data[id]
                imagename = data[id]["imagename"]
                img_path = os.path.join(base_path, "dataset/mm-vet/images", imagename)
                question = data[id]["question"]
                print(f"\n{id}")
                print(f"Image: {imagename}")

                example["num"] = num

                create_scene_graph_prompt = create_ccot_create_scene_graph_prompt(question)
                args.image_file= Image.open(img_path)
                args.image_path = img_path

                outputs = []
                scene_graphs = []

                for n in range(0, 3):
                    args.query = create_scene_graph_prompt
                    scene_graph = MLLM_generate(args)
                    args.query = create_ccot_answer_with_scene_graph_prompt(scene_graph, question)
                    output = MLLM_generate(args)
                    scene_graphs.append(scene_graph)
                    outputs.append(output) 
                example["scene_graphs"] = scene_graphs
                example["outputs"] = outputs

                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)
            
        elif args.reasoning == "ddcot":
            for i in range(begin_num, len(data)):
                id = f"v1_{i}"
                example = data[id]
                imagename = data[id]["imagename"]
                img_path = os.path.join(base_path, "dataset/mm-vet/images", imagename)
                question = data[id]["question"]
                print(f"\n{id}")
                print(f"Image: {imagename}")
                
                example["num"] = num
                
                create_subquestions_prompt = create_ddcot_create_subquestions_prompt(question)
                args.image_file= Image.open(img_path)
                args.image_path = img_path
            
                outputs = []
                subquestion_answerses = []
                    
                for n in range(0, 3):
                    args.query = create_subquestions_prompt
                    subquestion_answers = MLLM_generate(args)
                    args.query = create_ddcot_answer_with_subquestions_prompt(subquestion_answers, question)
                    output_ddcot = MLLM_generate(args)
                    subquestion_answerses.append(subquestion_answers)
                    outputs.append(output_ddcot)
                example["subquestion_answerses"] = subquestion_answerses
                example["outputs"] = outputs

                example["usage"] = get_usage(args.model)
                logs.append(example)
                num += 1
                record(args.dataset, args.model, args.reasoning, content = logs, split = args.split, lecture = args.lecture)