import os
from personality_tests import personality_assessment

from characters import character_info, character_labels
from utils import logger_main as logger
from utils import load_json_file
config = load_json_file("./config.json")

import pdb  
from utils import torch_gc
import argparse
os.environ["OPENAI_API_KEY"] = config["openai_apikey"]
questionnaires = config['questionnaires']
eval_method_map = {
    'self_report': 'choose',
    'self_report_cot': 'choosecot',
    'expert_rating': 'interview_assess_batch_anonymous',
    'expert_rating_collective': 'interview_assess_collective_anonymous',
    'option_conversion': 'interview_convert',
    'dimension_option_conversion': 'interview_convert_adjoption_anonymous'
}

parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--agent_llm', default='gpt-3.5', help='Agent LLM')
parser.add_argument('--eval_mode', default=False, choices=[True,False], help='Do Evaluation')
parser.add_argument('--eval_method', default='expert_rating', choices=eval_method_map.keys(), help='Evaluation method')
parser.add_argument('--eval_llm', default='gpt-3.5', choices=['gpt-4', 'gpt-3.5', 'gemini'], help='LLM for Evaluation')
parser.add_argument('--repeat_times', type=int, default=1, help='Number of experiment repeat times')


# 解析参数
args = parser.parse_args()
eval_mode = args.eval_mode
eval_method = eval_method_map.get(args.eval_method, args.eval_method)
eval_llm = args.eval_llm
repeat_times = args.repeat_times
agent_llm = args.agent_llm

characters = character_info.keys()
agent_types = ['RoleLLM', 'ChatHaruhi']
        
# there is a bug in transformer when interleave with luotuo embeddings and bge embeddings, which may sometimes cause failure. To minimize the change of embeddings, we run haruhi and rolellm characters separately.

results = {}
for character in characters:
    logger.info(f'Start Collecting data on {character}...')
    
    for questionnaire in questionnaires:
        logger.info(f'Processing on {questionnaire}...')
        for agent_type in agent_types:
            
        
        #for agent_type in [ a for a in character_info[character]['agent'] if a in agent_types]:
            if not agent_type in character_info[character]['agent']: continue
            if character == 'Sheldon-en' and agent_type == 'RoleLLM': continue 
            result = personality_assessment(
                character, agent_type, agent_llm, 
                questionnaire, eval_method, eval_llm, repeat_times=repeat_times,if_multiround=0,eval_mode=eval_mode)
            
            
            results[(character, agent_type)] = result 
    if not eval_mode:
        continue

    logger.info('Questionnaire: {}, Eval Method: {}, Repeat Times: {}, Agent LLM: {}, Eval LLM: {}'.format(questionnaire, eval_method, repeat_times, agent_llm, eval_llm))   

    from utils import avg

    personality_consistency = {} 

    for analysis_key in result['analysis'].keys():
        analysis_values = [ v['analysis'][analysis_key] for v in results.values()]
        analysis_value = avg(analysis_values)
        
        logger.info('Analyzing {}: {:.4f}'.format(analysis_key, analysis_value))
        personality_consistency[analysis_key] = analysis_value

    preds = { rpa: {dim: result['dims'][dim]['all_scores'] for dim in result['dims']} for rpa, result in results.items()}

    if questionnaire in ['BFI', '16Personalities']:
        label_settings = ['annotation', 'pdb']
        labels_pdb = { rpa: {dim: character_labels['pdb'][rpa[0]][questionnaire][dim] for dim in result['dims']} for rpa, result in results.items()} 
    else:
        label_settings = ['annotation']
        labels_pdb = { rpa: {dim: character_labels['annotation'][rpa[0]][questionnaire][dim] for dim in result['dims']} for rpa, result in results.items()} 

    for label_setting in label_settings:
        labels = { rpa: {dim: character_labels[label_setting][rpa[0]][questionnaire][dim] for dim in result['dims']} for rpa, result in results.items()} #e.g. { "score": 65.88130032806441, "type": "H"}


        from personality_tests import calculate_measured_alignment

        measured_alignment = calculate_measured_alignment(preds, labels, questionnaire, labels_pdb=labels_pdb)                        
        
        single_acc = measured_alignment['all']['single_acc']['all']
        single_mse = measured_alignment['all']['single_mse']['all']
        single_mae = measured_alignment['all']['single_mae']['all']
        full_acc = measured_alignment['all']['full_acc']
        
        
        
        logger.info('Alignment {}: Single Acc: {:.4f}, Single MSE: {:.4f}, Single MAE: {:.4f}, Full Acc: {:.4f}'.format(label_setting.upper()[:3], single_acc, single_mse, single_mae, full_acc))

            
                                            
                            
                            

                        
                        
                            
                            
                                
                
                    
                
                

        
            
        
