import json
import re
import os 
from collections import Counter
import random


base_path = "xxx"


def record(dataset, model, reasoning, path = None, content = None, split = "test", lecture = False):
    if path == None:
        if lecture:
            if not os.path.exists(os.path.join(base_path, "outputs_with_lectures")):
                os.mkdir(os.path.join(base_path, "outputs_with_lectures"))
            path = os.path.join(base_path,  "outputs_with_lectures/" + dataset + "_" + split + "_" + model + "_" + reasoning + ".json")
        else:
            if not os.path.exists(os.path.join(base_path, "outputs_none_lectures")):
                os.mkdir(os.path.join(base_path, "outputs_none_lectures"))
            path = os.path.join(base_path,  "outputs_none_lectures/" + dataset + "_" + split + "_" + model + "_" + reasoning + ".json")
    with open(path, "w") as f:
        json.dump(content, f, indent = 4)


def read_record(dataset, model, reasoning, path = None, split = "test", lecture = False):  
    if path == None:
        if lecture:
            path = os.path.join(base_path, "outputs_with_lectures/" + dataset + "_" + split + "_" + model + "_" + reasoning + ".json")
        else:
            path = os.path.join(base_path, "outputs_none_lectures/" + dataset + "_" + split + "_" + model + "_" + reasoning + ".json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.loads(f.read())
    else:
        return []
        
        
def read_json_from_path(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.loads(f.read())
    else:
        return []


def write_json_from_path(path, content):
    with open(path, "w") as f:
        return json.dump(content, f, indent = 4)


def extract_answer(output = None, dataset = "ScienceQA"):
    assert output != None
    if type(output) == list:
        output = output[0]
    try:
        if dataset == "ScienceQA":
            a = re.search(r"he answer is (.)", output)
            if a != None:
                option = a.group(1)
                if option < "A" or option > "G":
                    pattern = r"\(([A-G])\)"
                    match = re.search(pattern, output)
                    
                    if match == None:
                        option = re.search(r"\*\*(.)", output).group(1)
                    else:
                        option = match.group(1)
                        if option < "A" or option > "G":
                            option = re.search(r"\*\*(.)", output).group(1)
            else:
                a = re.search(r"correct option is (.)", output)
                if a != None:
                    option = a.group(1)
                    if option < "A" or option > "G":
                        pattern = r"\(([A-G])\)"
                        match = re.search(pattern, output)
                        option = match.group(1)
                else:
                    a = re.search(r"correct answer is (.)", output)
                    if a != None:
                        option = a.group(1)
                        if option < "A" or option > "G":
                            pattern = r"\(([A-G])\)"
                            match = re.search(pattern, output)
                            option = match.group(1)
                    else:
                        pattern = r"\(([A-G])\)"
                        match = re.search(pattern, output)
                        option = match.group(1)
            #assert ord(option) >= 65 and ord(option) <= 71
    except:
        print("extract answer error!")
        print(output)
        option =  None
    return option


def find_most_common_elements(input_list):
    input_list = [input for input in input_list if input != None]
    if len(input_list) == 0:
        return None
    counter = Counter(input_list)
    max_count = max(counter.values())
    most_common_elements = [element for element, count in counter.items() if count == max_count]
    return most_common_elements, max_count
    
    
def calculate_consistency_acc(dataset, model, reasoning, lecture = None, split = "test", path = None, num = 1, agents = 3, rounds = 3):
    if reasoning in ["io", "ccot", "ddcot"]:
        outputs = read_record(dataset, model, split = split, lecture = lecture, reasoning = reasoning)
        right_num = 0
        length = len(outputs)
        for output in outputs:
            true_answer = output["answer"]
            answers = []
            for n in range(0, num):
                answers.append(extract_answer(output["outputs"][n]))
            most_common_elements, counts = find_most_common_elements(answers)
            answer = random.choice(most_common_elements)
            if true_answer == ord(answer) - 65:
                right_num += 1
        acc = right_num / length
        print(f"Consistency-{num} Accuracy on {dataset}: {right_num} / {length} = {acc}")
    elif reasoning == "DMAD":
        path = os.path.join(base_path, "outputs_DMAD", f"{dataset}_{split}_{model}_{agents}_{rounds}.json")
        outputs = read_json_from_path(path)
        right_num = [0 for i in range(rounds)]
        length = len(outputs)
        for i in range(len(outputs)):
            output = outputs[i]
            true_answer = output["answer"]
            answers = []
            for n in range(rounds):
                answers = [extract_answer(answer) for answer in output["answers_"][n]]
                most_common_elements, counts = find_most_common_elements(answers)
                answer = random.choice(most_common_elements)
                if true_answer == ord(answer) - 65:
                    right_num[n] += 1
        for i in range(rounds):
            acc = right_num[i] / length 
            print(f"Accuracy of the {i}th round of DMAD on {dataset}: {right_num[i]} / {length} = {acc}")


def align_mmvet(model, reasoning, agents = 3, rounds = 3):
    path = os.path.join(base_path, "mm_vet_jsons")
    if not os.path.exists(path):
        os.mkdir(path)
    if reasoning in ["io", "ccot", "ddcot"]:
        outputs = read_record("mm-vet", model, reasoning)
        logs = {}
        for i in range(len(outputs)):
            name = f"v1_{i}"
            output = outputs[i]["outputs"][0]
            logs[name] = output
        write_json_from_path(os.path.join(path, f"{model}_{reasoning}.json"), logs)
        print(f"Results on MM-Vet are saved in {os.path.join(path, f'{model}_{reasoning}.json')}")
    elif reasoning == "DMAD":
        outputs = read_json_from_path(os.path.join(base_path, "outputs_DMAD", f"mm-vet_test_{model}_{agents}_{rounds}.json"))
        logs = [{} for i in range(rounds)]
        for i in range(len(outputs)):
            name = f"v1_{i}"
            for j in range(rounds):
                logs[j][name] = outputs[i]["eval_answer"][j]
        for j in range(rounds):
            write_json_from_path(os.path.join(path, f"{model}_{reasoning}_{agents}_{j+1}.json"), logs[j])
            print(f"Results of the {j+1}th round on MM-Vet are saved in {os.path.join(path, f'{model}_{reasoning}_{agents}_{j+1}.json')}")
