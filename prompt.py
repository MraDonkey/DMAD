prompt_ddcot_subquestions = '''Given the image, question and options, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

Sub-answers:
1. <sub-answer 1>
2. <sub-answer 2>
...
'''

prompt_ddcot_answer_with_subquestions = '''The problem can be deconstructed down to sub-questions. 
{subquestion_answers}
'''

prompt_ccot_make_scene_graph = '''For the provided image and its associated question, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Obect relationships that are relevant to answering the question.

Just generate the scene graph in JSON format. Do not say extra words.
'''

prompt_ccot_answer_with_scene_graph = '''The scene graph of the image in JSON format:
{scene_graph}

Use the image and scene graph as context and answer the following question.\n
'''

prompt_eval_answers = '''Here are some candidate answers using different methods. 
1. [
Directly answer the question.
{answer1}
]

2. [
First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{answer2}
]

3. [
First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{answer3}
]

Compare these candidate answers and their solving processes to reflect. Please choose the best candidate answer. You should only answer the number (1, 2 or 3) of candidate answers. If all the candidate answers above are incorrect, you should answer the number "4" only.
'''

prompt_eval_json_answers = '''Here are some candidate answers using different methods. 
1. [
Directly answer the question.
{answer1}
]

2. [
First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{answer2}
]

3. [
First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{answer3}
]
'''

prompt_ccot_solution = '''
First, get the scene graph of the image in JSON format:
{scene_graph}
Then, use the image and scene graph as context to answer the question.
{answer}
'''

prompt_ddcot_solution = '''
First, deconstruct the question down to sub-questions. 
{subquestion_answers}
Then, accord to the sub-questions and sub-answers to answer the question.
{answer}
'''

choose_one_option_prompt = '''Only one option is correct. Please choose the right option and explain why you choose it. You must answer in the following format. For example, if the right answer is A, you should answer: 
The answer is A. 
Because ...
'''