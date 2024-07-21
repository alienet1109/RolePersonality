# Capturing Minds, Not Just Words: Enhancing Role-Playing Language Models with Personality-Indicative Data

<p align="center">
🔔 <a href="https://github.com/alienet1109/RolePersonality" target="_blank">Code</a> • 📃 <a href="https://arxiv.org/abs/2406.18921" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/datasets/alienet/RolePersonality" target="_blank">Dataset</a> <br>
</p>

## Abstract
Role-playing agents (RPA) have been a popular application area for large language models (LLMs), attracting significant interest from both industry and academia.While existing RPAs well portray the characters' knowledge and tones, they face challenges in capturing their minds, especially for small role-playing language models (RPLMs). In this paper, we propose to enhance RPLMs via personality-indicative data. Specifically, we leverage questions from psychological scales and distill advanced RPAs to generate dialogues that grasp the minds of characters. Experimental results validate that RPLMs trained with our dataset exhibit advanced role-playing capabilities for both general and personality-related evaluations.

## Getting Started
```
git clone https://github.com/alienet1109/RolePersonality.git
cd RolePersonality
```
Edit `config.json` to specify target characters and target questionnaires. Attach OpenAI API key if using gpt-3.5 or gpt-4 as agent model. \
If the target character is not in `./data/characters.json`, please add the information into `./data/characters.json` following the format of existing characters. Notice that the character must be supported by ChatHaruhi or RoleLLM. \

Then run the command.
```
python run_experiments.py --agent_llm gpt-3.5 
```

## Evaluation
Please download [RoleBench Dataset](https://huggingface.co/datasets/ZenMoore/RoleBench) and put it into `/data` first.
```
cd data
git clone https://huggingface.co/datasets/ZenMoore/RoleBench
```
Edit `config.json` to specify target test characters. \
If the target character is not in `./data/characters.json`, please add the information into `./data/characters.json` following the format of existing characters. \
If the default prompt for your target character is not in `./data/characters_prompts.json`, please add it into `./data/characters_prompts.json` following the format of existing characters. Notice that the character must be supported by RoleLLM.\
Then run the command.
```
python evaluate.py --eval_method Rouge-L --target_llm gpt-3.5
```

## Citation
If you find our paper inspiring and have utilized it in your work, please cite our paper.
```
@misc{ran2024capturingmindsjustwords,
      title={Capturing Minds, Not Just Words: Enhancing Role-Playing Language Models with Personality-Indicative Data}, 
      author={Yiting Ran and Xintao Wang and Rui Xu and Xinfeng Yuan and Jiaqing Liang and Yanghua Xiao and Deqing Yang},
      year={2024},
      eprint={2406.18921},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18921}, 
}
```

## Contact
alienet1109@163.com