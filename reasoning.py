from prompt import *
from Dataset import *


def create_ddcot_create_subquestions_prompt(question, options = None, lecture = None, context = None):
    prompt = 'Question:\n' + question + '\n\n'
    if options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)
    prompt += prompt_ddcot_subquestions
    return prompt


def create_ddcot_answer_with_subquestions_prompt(subquestion_answers, question, options = None, lecture = None, context = None):
    prompt = 'Question:\n' + question + '\n\n'
    if options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)

    prompt += prompt_ddcot_answer_with_subquestions.format(subquestion_answers = subquestion_answers)
    if options != None:
        prompt += '''
        According to the sub-questions and sub-answers, give your option of the problem. Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
        The answer is A. 
        Because ...
        '''
    else:
        prompt += 'Give your answer of the question according to the sub-questions and sub-answers. Just answer the original question, do not say other extra words.'
    return prompt


def create_ccot_create_scene_graph_prompt(question, options = None, lecture = None, context = None):
    prompt = 'Question:\n' + question + '\n\n'
    if options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)
    prompt += prompt_ccot_make_scene_graph
    return prompt


def create_ccot_answer_with_scene_graph_prompt(scene_graph, question, options = None, lecture = None, context = None):
    prompt = prompt_ccot_answer_with_scene_graph.format(scene_graph = scene_graph)
    prompt = 'Question:\n' + question + '\n\n'
    if options != None:
        prompt += create_context(context)
        prompt += create_options(options)
        prompt += create_lecture(lecture)
        prompt += '''Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
    The answer is A. 
    Because ...
    '''
    return prompt


def create_eval_prompt(answers, scene_graph, subquestion_answers, question, options = None, lecture = None, context = None):
    prompt = 'Question:\n' + question + '\n\n'
    prompt += create_context(context)
    prompt += create_options(options)
    prompt += create_lecture(lecture)
    prompt += prompt_eval_answers.format(scene_graph = scene_graph, subquestion_answers = subquestion_answers, answer1 = answers[0], answer2 = answers[1], answer3 = answers[2])
    return prompt
