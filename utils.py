import pdb 
import os
import re 
import random 
import openai
import json
import logging
import time  
import jsonlines 
import requests 
import io
import pickle
import __main__

quiet = True
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('log.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 设置日志级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
if not quiet:
	logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
console_handler.setFormatter(formatter)
 
logger_main = logging.getLogger(__name__ + '_main')
file_handler_main = logging.FileHandler(f'{__main__.__file__}.log', encoding='utf-8')
logger_main.setLevel(logging.INFO)
logger_main.addHandler(file_handler_main)
logger_main.addHandler(console_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler_main.setFormatter(formatter)

cache_sign = True

def load_json_file(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)
def load_jsonl_file(path):
    data = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data
def save_json_file(path,target):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False,indent=True)

def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 

def dict_to_string(dic):
    ans = ""
    for key in dic:
        ans += f"{key}: {dic[key]}\n"
    return ans
cache = None 
def cached(func):
	def wrapper(*args, **kwargs):		
		global cache
		cache_path = 'cache.pkl'
		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				cache = pickle.load(open(cache_path, 'rb'))  

		key = ( func.__name__, str(args), str(kwargs.items()))
		
		

		if (cache_sign and key in cache and cache[key] not in [None, '[TOKEN LIMIT]']):
			return cache[key]
		else:
			result = func(*args, **kwargs)
			if result != 'busy' and result != None:
				cache[key] = result
				pickle.dump(cache, open(cache_path, 'wb'))
				#safe_pickle_dump(cache, cache_path)
			return result

	return wrapper
def string2json(llm_response, nth_generation=0, **kwargs):
	llm_response = llm_response.strip("`")
	if llm_response.startswith('json'):
		llm_response = llm_response[4:]
	
	try:
		json_response = json.loads(llm_response)
	except:
		try:
			llm_response = llm_response[llm_response.find("{"):]
			llm_response = '\n'.join([line.split("//")[0] for line in llm_response.splitlines()])

			json_response = json.loads(llm_response)
		except:
			return False
	
	return json_response

def string2json_ensure_choice_format(llm_response, nth_generation=0, **kwargs):
	
	if not llm_response: 		
		return False
	
	if isinstance(llm_response, str):
		json_response = string2json(llm_response)
		if not json_response:
			json_response = string2json(llm_response.replace(' x,\n', ' "x",\n'))
	else:
		json_response = llm_response
		
	if json_response:
		# gemini's ift ability need further improvement. it can not output the format I want
		try:
			for idx, choice in json_response.items():
			# 如果choice是一个dict
				if choice == 'x': continue 
				
				if isinstance(choice, dict):
					choice = choice['choice']
				if isinstance(choice, str):
					choice = int(float(choice))

				json_response[idx] = choice
			
			return json_response
				
		except:
			logger.info('Not a valid json response for choice format')
			
			return False
				
	else:
		##import pdb; pdb.set_trace()
		
		return False

def string2json_ensure_keys(llm_response, nth_generation=0, **kwargs):
	if not llm_response: return False

	if isinstance(llm_response, str):
		json_response = string2json(llm_response)
	else:
		json_response = llm_response
		
	if json_response:
		input_json = json.loads(kwargs['inputs'])
		
		missing_keys = input_json.keys() - json_response.keys()
		if missing_keys: 
			# the json response does not contain all the keys in input_json
			if nth_generation < 10:
				#import pdb; pdb.set_trace()
				
				return False
			else:
				for k in missing_keys:
					json_response[k] = 'x'
				return json_response
		else:
			return json_response
	else:
		#import pdb; pdb.set_trace()
		
		return False


def avg(lst):
	return sum(lst)/len(lst)

def std(lst):
	import math
	avg_score = avg(lst)
	return math.sqrt(sum([(s-avg_score)**2 for s in lst])/ (len(lst)))

def find_colon_idx(response):
	colon_idx1 = response.find(':')
	colon_idx2 = response.find('：')

	if colon_idx1 > -1 and colon_idx2 > -1:
		colon_idx = min(colon_idx1, colon_idx2)
	else:
		colon_idx = max(colon_idx1, colon_idx2) 
	return colon_idx

import tiktoken
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
	"""Return the number of tokens used by a list of messages."""
	try:
		encoding = tiktoken.encoding_for_model(model)
	except KeyError:
		print("Warning: model not found. Using cl100k_base encoding.")
		encoding = tiktoken.get_encoding("cl100k_base")
	if model in {
		"gpt-3.5-turbo-0613",
		"gpt-3.5-turbo-16k-0613",
		"gpt-4-0314",
		"gpt-4-32k-0314",
		"gpt-4-0613",
		"gpt-4-32k-0613",
		}:
		tokens_per_message = 3
		tokens_per_name = 1
	elif model == "gpt-3.5-turbo-0301":
		tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
		tokens_per_name = -1  # if there's a name, the role is omitted
	elif "gpt-3.5-turbo" in model:
		print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
		return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
	elif "gpt-4" in model:
		print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
		return num_tokens_from_messages(messages, model="gpt-4-0613")
	else:
		raise NotImplementedError(
			f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
		)
	num_tokens = 0
	for message in messages:
		num_tokens += tokens_per_message
		for key, value in message.items():
			num_tokens += len(encoding.encode(value))
			if key == "name":
				num_tokens += tokens_per_name
	num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
	return num_tokens

def is_multiround(response):
	"""
	Function to check if a string contains multiple rounds of conversation with
	improved handling for different languages and punctuation marks.
	If it does, return the first sentence spoken by the first speaker.
	If not, return False.
	"""

	# Split the response into parts based on speaker change indicators
	# Including handling for different languages (e.g., English, Chinese)
	normalized_response = response.replace("：", ":")
	
	# Split the response based on the response marker
	parts = normalized_response.split(":")
	parts_list = []
	current_part = ''
	eos_list = [s for s in '”]」.!?。？！\n']

	for part in parts:
		if any([ l in current_part[-15:] for l in eos_list]):
			# indicate new utterence
			parts_list.append(current_part.strip(':'))
			current_part = ''
		current_part += part + ':'
	parts_list.append(current_part.strip(':'))
	parts = parts_list	
	
	# Check if there are multiple rounds of response
	if len(parts) > 2:  # More than one ":" indicates multiple rounds
		# Reconstruct the response up to the second speaker
		response_up_to_second_speaker = ":".join(parts[:2])
		
		# Identify the last sentence of the first speaker
		# Considering both English and Chinese sentence terminators
		sentence_endings = '”]」.!?。？！'
		for ending in sentence_endings:
			if ending in response_up_to_second_speaker:
				last_sentence_end_index = response_up_to_second_speaker.rfind(ending)
				first_speaker_text = response_up_to_second_speaker[:last_sentence_end_index+1]
				
				return first_speaker_text.strip('\n\t "“”「」[]')
				
		# If no sentence terminator was found, return the whole part as the first sentence
		return response_up_to_second_speaker.strip('\n\t "“”「」[]')
	else:
		return False

def is_multilanguage(query, response):
	"""
	Function to count the number of English words, Chinese characters, Korean characters, and Japanese Kana
	in a given sentence.
	"""

	sentence = query + response
	sentence = re.sub(r'\(.*?\)', '', sentence)
	
	
	# Regular expressions for matching characters of different languages
	
	english_words_pattern = re.compile(r'[A-Za-z]+')
	chinese_characters_pattern = re.compile(r'[\u4e00-\u9fff]')
	korean_characters_pattern = re.compile(r'[\uac00-\ud7af]')
	japanese_kana_pattern = re.compile(r'[\u3040-\u30ff\u31f0-\u31ff]+')
	chaotic_pattern = re.compile(r'[�]')
	
	# Counting matches
	english_words_count = len(english_words_pattern.findall(sentence))
	chinese_characters_count = len(chinese_characters_pattern.findall(sentence))
	korean_characters_count = len(korean_characters_pattern.findall(sentence))
	japanese_kana_count = len(japanese_kana_pattern.findall(sentence))
	chaotic_pattern_count = len(chaotic_pattern.findall(sentence))
	
	count_language = 0 
	if english_words_count > 5:
		count_language += 1
	for count in [chinese_characters_count, korean_characters_count, japanese_kana_count, chaotic_pattern_count]:
		if count > 0:
			count_language += 1
	
	#print(count_language)
	if count_language > 1 :
		return True
	else:
		return False

def not_into_character(response, another_speaker):
	def contains_ai(sentence):
		"""
		Function to check if a sentence contains the isolated keyword "AI" with word boundaries,
		including cases where "AI" might be followed by punctuation like "AI." but not part of another word.
		"""
		# Regular expression pattern to match isolated "AI" with word boundaries, including following punctuation
		ai_pattern = re.compile(r'\bAI\b\.?')
		
		# Search for the pattern in the sentence
		match = ai_pattern.search(sentence)
		
		# Return True if a match is found, False otherwise
		return bool(match) or ('language model' in sentence.lower()) or ('语言模型' in sentence.lower())
	
	def start_with_others(sentence, another_speaker):
		sentence = sentence.replace('：', ':')
		if ':' in sentence[:15]: 
			first_speaker = sentence.split(':')[0][:15] 
			if another_speaker in first_speaker:
				return True
			
		return False 
	
	if contains_ai(response) or start_with_others(response, another_speaker):
		return True
	else:
		return False

def contain_repeation(response):
	# 修正正则表达式，确保每个汉字和日语字符都被单独匹配
	def tokenize_words(text):
		# 修正正则表达式：单独匹配每个汉字和日语字符
		#pattern = r'\b\w+\b|[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]|\d'
		# pattern = r'\b\w+\b|[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]|\d|[",.:?!]'

		# tokens = re.findall(pattern, text)
		
		import regex
		pattern = r'\b\w+\b|[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]|\d|[\p{P}\p{S}]'
		tokens = regex.findall(pattern, text)


		# 分离“文本测试”和“こんにちは”中的每个字符
		tokens_expanded = []
		for token in tokens:
			if re.match(r'[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]', token):
				tokens_expanded.extend(list(token))
			else:
				tokens_expanded.append(token)
		return tokens_expanded


	def detect_repetitions(tokens, min_length=15, max_length=30, threshold=0.1):
		"""
		检测token序列中重复子串的情况。
		
		:param tokens: Token序列，列表形式。
		:param min_length: 要检测的最小子串长度。
		:param max_length: 要检测的最大子串长度。
		:param threshold: 判断为大量重复的阈值（重复子串比例）。
		:return: 是否包含大量重复子串的布尔值。
		"""
		total_length = len(tokens)
		repetitions = 0
		
		first_repeat_idx = 999999999999999
		first_start_idx = {}

		# 遍历所有可能的子串长度
		for length in range(min_length, min(max_length + 1, total_length + 1)):
			substr_count = {}
			# 滑动窗口遍历token序列
			for i in range(total_length - length + 1):
				substr = tuple(tokens[i:i + length])  # 使用元组使子串可哈希
				if substr_count.get(substr, 0) > 0 :
					if i - first_start_idx[substr] >= length:			
						first_repeat_idx = min(first_repeat_idx, i)
						
				else:
					first_start_idx[substr] = i

				substr_count[substr] = substr_count.get(substr, 0) + 1
				
				
			
			# 计算重复子串的比例
			repetitions += sum(count > 1 for count in substr_count.values())
			
		
		# 计算重复率
		repetition_rate = repetitions / total_length if total_length else 0
	
		
		if first_repeat_idx < 999999999999999:
			return tokens[:first_repeat_idx]
		else:
			return False


	def concatenate_tokens(tokens):
		# 初始化空字符串用于拼接
		text = ""
		# 上一个token的类型，初始化为None
		last_type = None
		
		for token in tokens:
			# 检测当前token类型
			current_type = 'CJK' if re.match(r'[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]', token) else 'Other'
			import string
			if token in string.punctuation:
				current_type = 'P'
			
			# 如果当前token和上一个token都不是CJK（即汉字或日语字符），则在它们之间添加空格
			if last_type in ['Other', 'P'] and current_type == 'Other':
				text += " " + token
			else:
				text += token
				
			# 更新上一个token的类型
			last_type = current_type
		
		return text

	def find_long_letter_substrings(s):
		# Regular expression to match substrings of letters (a-Z and A-Z) with a length of at least 100
		pattern = r'[a-zA-Z]{100,}'
		# Find all matches
		matches = re.findall(pattern, s)
		return matches

	repeat_sign = False 

	_ = find_long_letter_substrings(response)
	if _:
		for substr in _:
			response = response.replace(substr, substr[:20])
		repeat_sign = True

	# 重新调用修正后的分词函数
	tokens = tokenize_words(response)

	_ = detect_repetitions(tokens)

	if _:
		response = concatenate_tokens(_)
		repeat_sign = True
	
	if repeat_sign:
		return response
	else:
		return False

def truncate(response):
	from langdetect import detect
	import re
	def count_chinese_characters(text):
		# 使用正则表达式匹配所有汉字字符
		chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
		return len(chinese_chars)
			
	if count_chinese_characters(response) > len(response) * 0.05:
		lang = 'zh'
	else:
		lang = 'en'
	
	if lang == 'zh':
		# 如果是中文，保留前300个字符
		return response[:500]
	else:
		# 如果是英文，用空格分词，然后保留前300个词
		words = response.split(' ')
		words = words[:500]
		return ' '.join(words)


if __name__ == '__main__':
	pass

	

	
		


