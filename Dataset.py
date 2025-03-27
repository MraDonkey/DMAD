from datasets import load_dataset
import os
import json

from record import base_path


def read_dataset(name, split):
    if name == "ScienceQA":
        data = load_dataset("derek-thomas/ScienceQA", split = split)
    elif name == "mm-vet":
        meta_data = os.path.join(base_path, "dataset/mm-vet/mm-vet.json")
        with open(meta_data, "r") as f:
            data = json.load(f)
    return data 


def create_options(options):
    if options != None:
        letters = ["(A) ", "(B) ", "(C) ", "(D) ", "(E) ", "(F) ", "(G) "]
        strs = "Options:\n"
        for i in range(len(options)):
            strs += letters[i]
            strs += options[i]
            strs += "\n"
        strs += "\n"
    else:
        strs = ""
    return strs


def create_lecture(lecture = None):
    strs = ""
    if lecture != None:
        strs = "Lecture:\n" + lecture + "\n\n"
    return strs


def create_context(context = None):
    strs = ""
    if context != None and context != "":
        strs = "Context:\n" + context + "\n\n"
    return strs


def create_prompt(question, options = None, context = None, lecture = None, if_options = True, post = True):
    prompt = "Question:\n" + question + "\n\n"
    if if_options and options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)
        postfix = '''Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
The answer is A. 
Because ...
'''        
        if post:
            prompt += postfix
    return prompt
    
    