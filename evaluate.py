import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts import *
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from utils import dict_to_string,torch_gc,load_json_file,load_jsonl_file,save_json_file,logger
import json
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
from ChatHaruhi.ChatHaruhi import get_models
config = load_json_file("./config.json")
os.environ["OPENAI_API_KEY"] = config["openai_apikey"]
questionnaires = config['questionnaires']
llm = None

parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--eval_method', type=str, default='Rouge-L', choices=['Rouge-L','ranking','5-dim'])
parser.add_argument('--target_llm', default='mistral', help='LLM for Evaluation')
parser.add_argument('--eval_llm', default='gpt-3.5', help='LLM for Evaluation')


class Evaluator():
    def __init__(self) -> None:
        self.role_prompts = get_all_role_prompts()
        pass
    
    def _get_results_save_path(self,role_name, model_name="", type = "5_dim"):
        if type == "5_dim":
            path = f"./{type}/{model_name}/{role_name}.json"
        elif type == "ranking":
            path = f"./{type}/{role_name}.json"
        elif type == "rougeL":
            path = f"./{type}/{role_name}.json"
       
            
        return path
    ##############################  RoleLLM Eval  ##############################
    def _roleLLM_get_QAs(self,role_code, type = "role-specific",data_dir = "./data/RoleBench",max_num = 300):
        qa_pairs = []
        role_name,language = role_code.split("-")
        if type == "role-specific":
        # 获取roleBench中的<Question>_<Ground truth>对
            if language == "en":
                file_name = f"instructions-eng/role-specific-{role_name}.jsonl"
                path = os.path.join(data_dir,file_name)
                origin_data = load_jsonl_file(path)
                for row in origin_data:
                    qa_pairs.append({
                        "instruction":row["instruction"],
                        "ground_truth":row["answer"]
                    })
            elif language == "zh":
                file_name = f"instructions-zh/role-specific-{role_name}.jsonl"
                path = os.path.join(data_dir,file_name)
                origin_data = load_jsonl_file(path)
                for row in origin_data:
                    qa_pairs.append({
                        "instruction":row["instruction"],
                        "ground_truth":row["answer"]
                    })
        elif type == "general":
            path = "./data/RoleBench/rolebench-eng/instruction-generalization/general/train.jsonl"
            origin_data = load_jsonl_file(path)
            for row in origin_data:
                if row["role"] == role_name:
                    qa_pairs.append({
                        "instruction":row["question"],
                        "ground_truth":row["generated"][0]
                    })
        return qa_pairs[:max_num]
    
    def rougeL_eval(self,role_code,model_name,question_type = "role-specific"):
        global llm
        data_save_path = f"./results/RoleBench/{question_type}/{model_name}/{role_code}.json"
        score_save_path = f"./results/RougeL/{question_type}/{model_name}/{role_code}.json"
        role_name =  role_code.split("-")[0]
        language = role_code.split("-")[1]
        if not llm:
            llm, tokenizer = get_models(model_name)
        print(f"Evaluating {model_name} on {role_name}")
        if os.path.exists(score_save_path):
            return
        if os.path.exists(data_save_path):
            qa_pairs = load_json_file(data_save_path) 
        else:
            from ChatHaruhi import ChatHaruhi
            if 'RoleLLM' in character_info[role_code]["agent"]:
            # RoleLLM
                character_agent = ChatHaruhi( role_from_hf = f'silk-road/ChatHaruhi-from-RoleLLM/{character_info[role_code]["agent"]["RoleLLM"]}', 
                                         llm_name = model_name, embedding = 'bge_en',llm=llm)
                character_agent.role_name = 'RoleLLM/' + character_info[role_code]["agent"]["RoleLLM"]
            elif 'ChatHaruhi' in character_info[role_code]["agent"]:
            # Haruhi
                character_agent = ChatHaruhi(role_name = role_name, llm = model_name)
                character_agent.role_name = 'ChatHaruhi/' + character_info[role_code]["agent"]["ChatHaruhi"]
            
            experimenter = get_experimenter(role_code)
            qa_pairs = self._roleLLM_get_QAs(role_code,type=question_type)
            # 生成并储存模型回答
            for i,pair in tqdm(enumerate(qa_pairs)):
                character_agent.dialogue_history = []
                response = character_agent.chat(role = experimenter, text = pair['instruction'])
                qa_pairs[i]['response'] = response
            save_json_file(data_save_path,qa_pairs)
        gt_lis = [pair['ground_truth'] for pair in qa_pairs]
        rs_lis = [pair['response'] for pair in qa_pairs]    
        # 全部完成后计算rougeL分数
        score_dic = self._calculate_rougeL(gt_lis,rs_lis)
        final_result = {
            'Rouge-L': score_dic,
            'details': qa_pairs
        }
        save_json_file(score_save_path,final_result)
        
    def _calculate_rougeL(self,model_answers,ground_truths):
        from rouge import Rouge 
        result = Rouge().get_scores(model_answers, ground_truths)
        score_dic = {"r":0,"p":0,"f":0}
        for score in result:
            for key in score_dic:
                score_dic[key] += score['rouge-l'][key]
        for key in score_dic:
            score_dic[key] /= len(result)
        return score_dic

    def _get_roleLLM_data(self,role_code,model_names,data_dir = "./results/RougeL/role-specific"):
        '''
        [{model1_name:"",model2_name:"",model3_name:"",question:""},{},...]
        '''
        qa_model_pairs = []
        qa_pairs_dict = {}
        for model_name in model_names:
            name = f"{model_name}/{role_code}.json"
            path = os.path.join(data_dir,name)
            data = load_json_file(path)
            qa_pairs_dict[model_name] = data["details"]
        for i in range(len(qa_pairs_dict[model_name])):
            temp = {}
            for model_name in model_names:
                temp[model_name] = qa_pairs_dict[model_name][i]['response']
            temp['question'] = qa_pairs_dict[model_name][i]['instruction']
            qa_model_pairs.append(temp)
        return qa_model_pairs
    def gpt_ranking(self,role_code,model_names):
        '''
        Questions from RoleLLM
        '''
        def roleLLM_get_QAs(role_code, type = "role-specific",data_dir = "./data/RoleBench"):
            # 获取roleBench中的<Question>_<Ground truth>对
            qa_pairs = {}
            role_name,language = role_code.split("-")
            if language == "en" and type == "role-specific":
                file_name = f"instructions-eng/role-specific-{role_name}.jsonl"
                path = os.path.join(data_dir,file_name)
                origin_data = load_jsonl_file(path)
                for row in origin_data:
                    qa_pairs[row["instruction"]]=row["answer"]
            if language == "zh" and type == "role-specific":
                file_name = f"instructions-zh/role-specific-{role_name}.jsonl"
                path = os.path.join(data_dir,file_name)
                origin_data = load_jsonl_file(path)
                for row in origin_data:
                    qa_pairs[row["instruction"]]=row["answer"]
            return qa_pairs
        qa_dic = roleLLM_get_QAs(role_code)
        
        role_name = role_code.split("-")[0]
        prompt = ChatPromptTemplate.from_messages([
            ("user", prompt_ranking_en)
        ])
        llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
        output_parser = JsonOutputParser()
        chain_ranking = prompt | llm | output_parser
        role_profile = prompt_to_profile(role_name,self.role_prompts[role_name])
        
        print(f"Evaluating ranking on {role_name}...")
        save_dir = "./results/ranking/"
        model_dir = "-".join(model_names) + "/"
        assessment = {'avg_rank':{model_name: 0 for model_name in model_names},
                'avg_win_rate':{model_name: 0 for model_name in model_names},}
        save_path = os.path.join(save_dir,model_dir,role_name+".json")
        if os.path.exists(save_path):
            assessment.update(load_json_file(save_path))
        
        rank_dict = {model_name: 0 for model_name in model_names}
        win_rate_dict = {model_name: 0 for model_name in model_names}
        details = []
        qa_model_pairs = self._get_roleLLM_data(role_code,model_names)
        
        for pair in tqdm(qa_model_pairs):
            
            # output = [{"model":"mistral","reason":"","rank":3},{"model":"mistralIC","reason":"","rank":2}]
            # output为dic组成的list，dic = {"model":model_name,"":,"reason":reason,"rank":rank}
            try:
                output = chain_ranking.invoke({
            "role_name": role_name,
            "question_dict": pair['question'],
            "role_description_and_catchphrases":role_profile,
            "list_model_answer_dict": dict_to_string({model_name:trim_text_to_n_words(pair[model_name],200) for model_name in model_names})
            })
                detail = []
                for dic in output:
                    rank_dict[dic['model']] += int(dic['rank'])
                    if int(dic['rank']) == 1:
                        win_rate_dict[dic['model']] += 1
                    dic["question"] = pair["question"]
                    dic["answer"] = pair[dic['model']]
                    detail.append(dic)
                details.append(detail)
            except Exception as e:
                if e == KeyboardInterrupt:
                    break
                else:
                    print(e)
                    continue
        for model_name in model_names:
            rank_dict[model_name] /= len(details)
            win_rate_dict[model_name] = win_rate_dict[model_name] / len(details) * 100
        assessment= {
            'avg_rank':rank_dict,
            'avg_win_rate':win_rate_dict,
            'details':details
        }
        save_json_file(save_path,assessment)
            
    
    ##############################  CharacterLLM Eval  ##############################
    
    def gpt_5_dim(self,role_code,model_name):
        def extract_score(output):
            rows = output.split("\n")
            lis = rows[-1].split(" ")  + rows[-2].split(" ") + rows[0].split(" ") 
            for word in lis:
                if word.isnumeric():
                    return float(word)

        def get_roleLLM_data(role_code,model_name,data_dir = "./results/RougeL/role-specific"):
            name = f"{model_name}/{role_code}.json"
            path = os.path.join(data_dir,name)
            data = load_json_file(path)
            qa_pairs = data["details"]

            return qa_pairs[:]
        role_name = role_code.split("-")[0]
        print(f"Evaluating 5-dim performance of {model_name} on {role_name}...")
        prompts_5_dim = {"memorization":prompt_eval_memorization,
                         "personality":prompt_eval_personality,
                         "values":prompt_eval_values,
                         "stability":prompt_eval_stability,
                         "hallucination":prompt_eval_hallucination,}
        save_dir = "./results"
        save_path = os.path.join(save_dir,self._get_results_save_path(role_code,model_name))
        
        role_profile = prompt_to_profile(role_name,self.role_prompts[role_name])
        llm = ChatOpenAI(temperature=0)
        output_parser = StrOutputParser()
        assessment = {'score':{}}
        
        if os.path.exists(save_path):
            assessment.update(load_json_file(save_path))
            
        for prompt_name in prompts_5_dim:
            print(f"Evaluating on {prompt_name}...")
            dim_score = 0
            dim_scale = 0   
            prompt = ChatPromptTemplate.from_template(prompts_5_dim[prompt_name])
            chain =  prompt | llm | output_parser
            if prompt_name not in assessment:
                assessment[prompt_name] = {}
            qa_pairs = get_roleLLM_data(role_code,model_name)
            details = []
            total_score = 0
            for pair in tqdm(qa_pairs):    
                while True:
                    output = chain.invoke({
                    "agent_name":role_name,
                    "agent_context":trim_text_to_n_words(role_profile,200),
                    "interactions":trim_text_to_n_words(construct_interactions(role_code,pair),200)
                })
                    score = extract_score(output)
                    if isinstance(score,float): 
                        break
                    else:
                        print(output)
                        
                total_score += score
                details.append({
                    "question": pair['instruction'],
                    "response": pair['response'],
                    "evaluation": output,
                    "score": score
                })
            assessment[prompt_name] = {
                'avg_score': total_score / len(qa_pairs),
                'profile':role_profile,
                'details': details
                }
            dim_score += total_score
            dim_scale += len(qa_pairs)
            assessment[prompt_name]['avg_score'] = dim_score / dim_scale
            save_json_file(save_path,assessment)
                
        for prompt_name in prompts_5_dim:
            assessment['score'][prompt_name] = assessment[prompt_name]['avg_score']
        save_json_file(save_path,assessment)
       
      
def get_role_lines(role_code):
    #获取角色标志性台词用于评估
    pass  
def trim_text_to_n_words(text,n=5):
    lis = []
    num_words = 0
    sentences = text.split(".")
    for sentence in sentences:
        temp_lis = sentence.split(" ")
        if num_words + len(temp_lis) < n:
            lis += temp_lis + ['.']
            num_words += len(temp_lis)
        else:
            break
    return " ".join(lis)
        


def construct_interactions(role_code, dic):
    role_name = role_code.split("-")[0]
    return f"{get_experimenter(role_code)}: {dic['instruction']}\n{role_name}: {dic['response']}\n"

def prompt_to_profile(role_name, prompt):
    lis = prompt.split("\n")
    i = 0
    while i<=len(lis)-1:
        i += 1
        if "You must know" in lis[i]:
            break
    return f"{role_name} is " + " ".join(lis[i+1:])

def get_all_role_prompts(path = "./data/character_prompts.json"):
    role_prompts = load_json_file(path)
    return role_prompts


character_info = load_json_file("./data/characters.json")
def get_experimenter(role_code):    
    return character_info[role_code]["experimenter"]

if __name__ == "__main__":
    evaluator = Evaluator()
    args = parser.parse_args()
    eval_method = args.eval_method
    target_llm = args.target_llm
    eval_llm = args.eval_llm
    roles = config['eval_characters']
    if eval_method == 'Rouge-L':
        for role_code in roles: 
            evaluator.rougeL_eval(role_code,target_llm)
            
    elif eval_method == 'ranking':      
        for role_code in roles: 
            evaluator.gpt_ranking(role_code,model_names=[target_llm,"gpt-4"])
    elif eval_method == '5-dim':
        for role_code in roles: 
            evaluator.rougeL_eval(role_code,target_llm)
            evaluator.gpt_5_dim(role_code,target_llm)

        