prompt_ranking_cn = '''
你是一个角色扮演的效果对比助手，你会根据输出的角色特征和质量来对模型进行排名，然后使
用python dict list 输出结果。
User Prompt:
下 列 模 型 要 扮 演 的 角 色 是“{role_name}”。{role_name}的 角 色 描 述
是“{role_description_and_catchphrases}”。我需要根据下面两个原则对下列模型进行排名：
1. 哪一个的角色说话风格特征更加明显，说话更加符合角色描述，说话越有特色就越好；
2. 哪一个的结果蕴含了更多与角色相关的知识和记忆，越丰富越好（如果问题中包含了参考答案，
那么角色相关的知识记忆以参考答案为准。）
输入给各个模型的问题是：
{question_dict}
各个模型针对该问题的回答分别为：
{list_model_answer_dict}
现在请你根据上述两个原则，对各个模型进行排名。避免任何位置偏见，并确保模型回答的呈现顺
序不会影响你的决定。不要对模型的名字带有偏见。然后使用一个包含模型与其排名、这样排名的
理由的列表返回结果，也就是说，请务必使用如下格式返回结果：
[{“model”: <model-name>, “reason”: <rank-reason>, “rank”: <model-rank>}, {“model”: <model-name>,
“reason”: <rank-reason>, “rank”: <model-rank>}]
你的回答必须是一个有效的python 字典列表以保证我能够直接使用python 解析它，不要有多余的内
容！请给出尽可能准确的、符合大多数人直觉的排名。

'''
prompt_ranking_en = '''
System Instruction:
You are a role−playing performance comparison assistant. You should rank the models based on the role
characteristics and text quality of their responses. The rankings are then output using Python dictionaries and
lists.
User Prompt:
The models below are to play the role of ‘‘{role_name}’’. The role description of ‘‘{role_name}’’ is ‘‘{role_description_and_catchphrases}’’. 
I need to rank the following models based on the two criteria below:
1. Which one has more pronounced role speaking style, and speaks more in line with the role description.
The more distinctive the speaking style, the better.
2. Which one’s output contains more knowledge and memories related to the role; the richer, the better. (If
the question contains reference answers, then the role−specific knowledge and memories are based on the
reference answer.)
The question provided to each model is:
{question_dict}
The respective answers from the models to this question are:
{list_model_answer_dict}
Now, based on the above two criteria, please rank the models. Avoid any positional biases and ensure that the
order in which the responses are presented does not influence your decision. Do not favor certain model
names.
Then, use a list containing the model’s name, its rank, and the reason for its ranking to return the results, i.e.,
please ensure to use the following format to return the results:
[{{‘‘model’’: <model−name>, ‘‘reason’’: <rank−reason>, ‘‘rank’’: <model−rank>}}, {{‘‘model’’: <model−
name>, ‘‘reason’’: <rank−reason>, ‘‘rank’’: <model−rank>}}]
Your answer must be a valid Python list of dictionaries to ensure I can directly parse it using Python. Do not
include any extraneous content! Please provide a ranking that is as accurate as possible and aligns with the
intuition of most people.
'''
prompt_ranking_en_1 = '''
System Instruction:
You are a role−playing performance comparison assistant. You should rank the models based on the role
characteristics and text quality of their responses. The rankings are then output using Python dictionaries and
lists.
User Prompt:
The models below are to play the role of ‘‘{role_name}’’. The role description of ‘‘{role_name}’’ is ‘‘{role_description_and_catchphrases}’’. 
I need to rank the following models based on the two criteria below:
1. Which one has more pronounced role speaking style, and speaks more in line with the role description.
The more distinctive the speaking style, the better.
2. Which one’s output contains more memories related to the role; the richer, the better. (If
the question contains reference answers, then the role−specific knowledge and memories are based on the
reference answer.)
3. The closer the answer is to the ground truth, the better.
4. If the answer is off topic (not closely related to the question), or formatted incorrectly (multi-rounds dialogue), it is a bad answer.
The question provided to each model is:
{question_dict}
The ground truth is:
{ground_truth}
The respective answers from the models to this question are:
{list_model_answer_dict}
Now, based on the above two criteria, please rank the models. Avoid any positional biases and ensure that the
order in which the responses are presented does not influence your decision. Do not favor certain model
names.
Then, use a list containing the model’s name, its rank, and the reason for its ranking to return the results, i.e.,
please ensure to use the following format to return the results:
[{{‘‘model’’: <model−name>, ‘‘reason’’: <rank−reason>, ‘‘rank’’: <model−rank>}}, {{‘‘model’’: <model−
name>, ‘‘reason’’: <rank−reason>, ‘‘rank’’: <model−rank>}}]
Your answer must be a valid Python list of dictionaries to ensure I can directly parse it using Python. Do not
include any extraneous content! Please provide a ranking that is as accurate as possible and aligns with the
intuition of most people.
'''
prompt_ranking_en_2 = '''
System Instruction:
You are a role−playing performance comparison assistant. You should rank the models based on the role
characteristics and text quality of their responses. The rankings are then output using Python dictionaries and
lists.
User Prompt:
The models below are to play the role of ‘‘{role_name}’’. The role description of ‘‘{role_name}’’ is ‘‘{role_description_and_catchphrases}’’. 
Your can refer to the role's original lines: ‘‘{original_line}’’
I need to rank the following models based on the two criteria below:
1. Which one has more pronounced role speaking style, and speaks more in line with the role description.
The more distinctive the speaking style, the better.
2. Which one’s output contains more memories related to the role; the richer, the better. (If
the question contains reference answers, then the role−specific knowledge and memories are based on the
reference answer.)
3. The closer the answer is to the ground truth, the better.
4. If the answer is off topic (not closely related to the question), or formatted incorrectly (multi-rounds dialogue), it is a bad answer.
The question provided to each model is:
{question_dict}
The ground truth is:
{ground_truth}
The respective answers from the models to this question are:
{list_model_answer_dict}
Now, based on the above two criteria, please rank the models. Avoid any positional biases and ensure that the
order in which the responses are presented does not influence your decision. Do not favor certain model
names.
Then, use a list containing the model’s name, its rank, and the reason for its ranking to return the results, i.e.,
please ensure to use the following format to return the results:
[{{‘‘model’’: <model−name>, ‘‘reason’’: <rank−reason>, ‘‘rank’’: <model−rank>}}, {{‘‘model’’: <model−
name>, ‘‘reason’’: <rank−reason>, ‘‘rank’’: <model−rank>}}]
Your answer must be a valid Python list of dictionaries to ensure I can directly parse it using Python. Do not
include any extraneous content! Please provide a ranking that is as accurate as possible and aligns with the
intuition of most people.
'''

prompt_eval_memorization = '''

You will be given responses written by an AI assistant mimicking the character {agent_name}. Your task is to rate the performance of {agent_name} using the
specific criterion by following the evaluation steps. Be as strict as possible. Below is the data:
***
[Profile]
{agent_context}

***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Factual Correctness (1-7): Is the response provides truthful and detailed facts about the character?
[Evaluation Steps]
1. Read through the interactions and identify the key points related to the character.
2. Read through the responses of the AI assistant and compare them to the profile. Check if the responses are consistent with the character’s profile, background, and
known facts about the character.
3. Check whether the responses provide detailed facts about the character or if they are generic responses that could apply to any character. Detailed responses are
more factual and contribute positively to the score.
4. Rate the performance of the AI on a scale of 1-7 for factual correctness, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria.
***
First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.
'''

prompt_eval_personality = '''
You will be given responses written by an AI assistant mimicking the character {agent_name}. Your task is to rate the performance of {agent_name} using the
specific criterion by following the evaluation steps. Be as strict as possible. Below is the data:
***
[Profile]
{agent_context}

***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Personality (1-7): Is the response reflects the personalities and preferences of the character?
[Evaluation Steps]
1. Read through the profile and write the personalities and preferences of the real character.
2. Read through the interactions and identify the personalities and preferences of the AI assistant.
3. After having a clear understanding of the interactions, compare the responses to the profile. Look for any consistencies or inconsistencies. Do the responses reflect
the character’s personalities and preferences?
4. Use the given scale from 1-7 to rate how well the response reflects the personalities and preferences of the character. 1 being not at all reflective of the character’s
personalities, and 7 being perfectly reflective of the character’s personalities.
***
First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.


'''
prompt_eval_values = '''
You will be given responses written by an AI assistant mimicking the character {agent_name}. Your task is to rate the performance of {agent_name} using the
specific criterion by following the evaluation steps. Be as strict as possible. Below is the data:
***
[Profile]
{agent_context}

***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Values (1-7): Is the response reflects the values and convictions of the character?
[Evaluation Steps]
1. Read through the profile and write the values and convictions of the real character.
2. Read through the interactions and identify the values and convictions of the AI assistant.
3. After having a clear understanding of the interactions, compare the responses to the profile. Look for any consistencies or inconsistencies. Do the responses reflect
the character’s values and convictions?
4. Use the given scale from 1-7 to rate how well the response reflects the values and convictions of the character. 1 being not at all reflective of the character’s values,
and 7 being perfectly reflective of the character’s values.
***
First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.
'''
prompt_eval_hallucination = '''
You will be given responses written by an AI assistant mimicking the character {agent_name}. Your task is to rate the performance of {agent_name} using the
specific criterion by following the evaluation steps. Be as strict as possible. Below is the data:
***
[Profile]
{agent_context}

***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Avoiding Hallucination (1-7): Is the response avoids to say things that the character do not know?
[Evaluation Steps]
1. Read through the interactions and identify the knowledge scope of the character.
2. Read through the responses of the AI assistant, find the evidence of knowledge used in the response.
3. Compare the evidence to the profile. Check if the responses are consistent with the character’s knowledge scope. If some knowledge contradicts to the character’s
identity, given a lower score. Otherwise, assign a higher score.
4. Rate the performance of the AI on a scale of 1-7 for Avoiding Hallucination, where 1 is the lowest and 7 is the highest based on the Evaluation Criteria.
***
First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.
'''

prompt_eval_stability = '''

You will be given responses written by an AI assistant mimicking the character {agent_name}. Your task is to rate the performance of {agent_name} using the
specific criterion by following the evaluation steps. Be as strict as possible. Below is the data:
***
[Profile]
{agent_context}

***
[Interactions]
{interactions}
***
[Evaluation Criterion]
Long-term Acting (1-7): Is the assistant maintain a good performance over the long interactions?
[Evaluation Steps]
1. Read through the given profile and background information to familiarize yourself with the context and details of the AI assistant named {agent_name}.
2. Review the interactions provided to see how {agent_name} responds to various prompts and queries. And evaluate the performance of acting query by query that
whether the response reflects the personalities and values of the character. Assign score for each turn.
3. Based on the above assigned scores, does {agent_name} keep actinig like character in the long-term? Evaluate the overall performance of the whole conversation
based on the score for each turn.
4. Rate the stability of {agent_name} on a scale of 1 to 7, with 1 being very poor and 7 being excellent.
***
First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the
outset. Then print the score on its own line corresponding to the correct answer. At the end, repeat just the selected score again by itself on a new line.
'''

prompts = {
    "general": {
        "background_template": '''You are an expert in Psychometrics, especially {}. I am conducting the {} test on someone. I am gauging his/her position on the {} dimension through a series of open-ended questions. For clarity, here's some background this particular dimension:
===
{}
===

My name is {}. I've invited a participant, {}, and we had many conversations in {}. I will input the conversations.

Please help me assess {}'s score within the {} dimension of {}. 
''',
    "two_score_output": '''You should provide the percentage of each category, which sums to 100%, e.g., 30% A and 70% B. 
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": {{ "{}": <percentage 1>, "{}": <percentage 2> }} (The sum of percentage 1 and percentage 2 should be 100%. Output with percent sign.) 
}}''',
    "one_score_output": '''You should provide the score of {} in terms of {}, which is a number between {} and {}. {} denotes 'not {} at all', {} denotes 'neutral', and {} denotes 'strongly {}'. Other numbers in this range represent different degrees of '{}'. 
Please output in the following json format:
===
{{
    "analysis": <your analysis based on the conversations>,
    "result": <your score>
}}'''
    },
}