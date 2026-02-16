import time

from rich import print
from openai import OpenAI
import random
from typing import Literal
import copy
from pathlib import Path
from collections import Counter
import numpy as np

from jarvisvla.inference import action_mapping, load_model, processor_wrapper
from jarvisvla.utils.file_utils import load_json_file

#################
# prompt
#################

INSTRUCTION_TEMPLATE = [
    "help me to craft a {}.",
    "craft a {}.",
    "Craft a {}.",
    "Could you craft a {} for me?",
    "I need you to craft a {}.",
    "Please craft a {} in the game.",
    "Craft me a {} quickly.",
    "Make sure to craft a {} for the task.",
    "Craft a {} so I can use it.",
    "Let’s craft a {} for this project.",
    "I need you to craft {} right now.",
]

class VLLM_AGENT:
    def __init__(self,checkpoint_path, base_url, api_key="EMPTY",
                 history_num=0,action_chunk_len=1, bpe=0,
                 instruction_type:Literal['simple','recipe','normal'] = 'normal',
                 temperature=0.5):
        
        self.LLM_backbone,self.VLM_backbone = load_model.load_visual_model(checkpoint_path=checkpoint_path)
        self.action_tokenizer = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone)
        
        self.prompt_library = load_json_file(Path(__file__).parent/"assets"/"instructions.json") # Pre-written instructions
        self.recipe_fold=Path(__file__).parent/"assets"/"recipes" # Folder storing all recipes
        self.recipes = dict()  # Collection of crafting recipes
        self.method_map = {
            True: "crafting table",
            False: "inventory",
        }
        
        self.processor_wrapper = processor_wrapper.ProcessorWrapper(None,model_name=self.VLM_backbone)
       
        self.actions = []
        self.action_chunk_len=action_chunk_len  # Return one action chunk at a time
        # For agent with memory
        self.history_num = history_num
        self.history = []
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        models = self.client.models.list()
        self.model = models.data[0].id
        
        self.temperature = temperature
        self.instruction_type = instruction_type
        
        self.tokenizer = None
        from transformers import AutoTokenizer
        if self.LLM_backbone in {"llama-3","llama-2","qwen2_vl"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,  
                trust_remote_code=True,
            )
            if self.LLM_backbone=="qwen2_vl" and len(self.tokenizer)==151657:
                import json
                with open("ultron/model/assets/special_token.json", "r") as file:
                    special_token = json.load(file)
                self.tokenizer.add_special_tokens({"additional_special_tokens":special_token})
    
    def reset(self):
        self.history = []
          
    def rule_based_instruction(self,env_prompt:str):
        item_name = env_prompt[11:].replace("_"," ")
        instruction_template = random.choice(INSTRUCTION_TEMPLATE)
        return instruction_template.format(item_name)
          
    def create_basic_instruction(self,env_prompt:str):
        instruction = "."
        instructions = self.prompt_library.get(env_prompt,{}).get("instruct")
        if instructions:
            instruction = np.random.choice(instructions)
        else:
            instruction = self.rule_based_instruction(env_prompt)

        if instruction.strip()[-1] != '.':
            instruction += ". \n"
        instruction += "\n"
        return instruction
        
    def get_recipe_item_name(self,ingredient:dict):
        item_name = ingredient.get("item") 
        if not item_name:
            item_name = ingredient.get("tag") 
        return item_name
        
    def create_recipe_prompt_from_library(self,item_name:str):
        if item_name in self.recipes:
            return self.recipes[item_name]
        recipe_path = self.recipe_fold/f"{item_name}.json"
        print(recipe_path)
        if not recipe_path.exists():
            self.recipes[item_name]= ""
            return ""
        recipe_file = load_json_file(recipe_path)
        recipe_type = recipe_file.get("type",None)
        
        prompt = ""
        if not recipe_type:
            self.recipes[item_name]= ""
            return ""
        elif recipe_type=="minecraft:crafting_shapeless":
            prompt+=f"\nYou will need the following ingredients: \n"
            ingredients = recipe_file.get("ingredients",None)
            ingredients_list = []
            for ingredient in ingredients:
                ingredient_name = self.get_recipe_item_name(ingredient)
                if not ingredient_name:
                    break
                ingredients_list.append(ingredient_name[10:].replace("_"," "))
            ingredients_dict = Counter(ingredients_list)
            for item,number in ingredients_dict.items():
                prompt += f"{number} {item}, "
            prompt += "\n"
        elif recipe_type == "minecraft:crafting_shaped":
            prompt+="\nArrange the materials in the crafting grid according to the following pattern: \n"
            patterns = recipe_file.get("pattern",None)
            if not patterns:
                return ""
            ingredients = recipe_file.get("key",{})
            ingredients_dict = {}
            for ingredient_mark,value in ingredients.items():
                ingredient_name = self.get_recipe_item_name(value)
                ingredients_dict[ingredient_mark] = ingredient_name[10:]
            prompt+="\n"
            for pattern_line in patterns:
                for pattern_mark in pattern_line:
                    ingredients_name = ""
                    if pattern_mark==" ":
                        ingredients_name = "air"
                    else:
                        ingredients_name = ingredients_dict.get(pattern_mark,"air")
                    prompt += f" {ingredients_name} |"
                if prompt[-1]=='|':
                    prompt = prompt[:-1] 
                    prompt += "\n"
            prompt +="\n"
        else:
            self.recipes[item_name]= ""
            return ""
        result_num = recipe_file.get("result",{}).get("count",1)
        prompt += f"and get {result_num} {item_name.replace('_',' ')}. \n"
        self.recipes[item_name]= prompt
        return prompt
        
    def create_recipe_prompt(self,env_prompt:str,method:str="crafting_table"):
        """Convert raw instruction text into a prompt"""
        prompt = ""
        # If using recipe book to craft
        if "recipe_book" in method:
            prompt += "\nUse the recipe book to craft. \n"
            return prompt
        # else
        recipe = self.prompt_library.get(env_prompt,{}).get("recipe")
        if recipe:
            prompt += "\nArrange the materials in the crafting grid according to the following pattern: \n"
            prompt += recipe[0]
        else:
            item_name = env_prompt.replace(" ","_").split(":")[-1]
            prompt += self.create_recipe_prompt_from_library(item_name,)
        return prompt
        
    def create_instruction(self,env_prompt,method):
        prompt =None
        if self.instruction_type == 'recipe':
            prompt = self.create_basic_instruction(env_prompt)
            recipe_prompt = self.create_recipe_prompt(env_prompt,method=method)

            prompt += recipe_prompt
        elif self.instruction_type == 'simple':
            prompt = self.create_thought(env_prompt) 
        elif self.instruction_type == 'normal':
            natural_text = env_prompt.replace("_"," ").replace(":"," ")
            prompt = random.choice(self.prompt_library.get(env_prompt,{"instruct":[natural_text]})["instruct"])
        else:
            raise ValueError(f"do not set the instruction class {self.instruction_type}")
        return prompt
        
    def create_thought(self,env_prompt):
        thought = copy.copy(self.prompt_library.get(env_prompt,{}).get("thought"))
        if not thought:
            thought = env_prompt.replace("item",str(1)).replace("_"," ").replace(":"," ")   #craft item xxx =》 craft 1 item
        thought += '. \n'
        return thought
        
    def forward(self,observations,instructions,verbos=False,need_crafting_table=False):
        if self.actions:
            if verbos:
                print(self.actions)
            # Use cached LLM output if we're returning cached actions
            if len(self.actions)>1:
                return self.actions.pop(0)
            else:
                action = self.actions[0]
                self.actions = []
                return action
        messages = []
        image = self.processor_wrapper.create_image_input(observations[0]) 
        method = self.method_map[need_crafting_table]
        prompts = []
        private_instruction = self.create_instruction(instructions[0],method=method)
        self.last_instruction = private_instruction
        if verbos:
            print(f"\n=== INSTRUCTION ===\n{private_instruction}\n===================")
        thought= self.create_thought(instructions[0]) if self.instruction_type =="recipe" else ""
        if verbos and thought:
            print(f"=== THOUGHT/RECIPE ===\n{thought}\n======================")

        if self.history_num:
            if not self.history: # If history is empty
                self.history = [(image,self.action_tokenizer.null_token(),copy.copy(thought),0)]*self.history_num
            new_history = [None]*self.history_num
            new_history[:-1] = self.history[1:]
            for hdx,(im, ac, past_thought,_) in enumerate(self.history):
                prompt_input = ""
                if self.instruction_type == 'recipe':
                    prompt_input = "\nthought: " + past_thought + "\nobservation: "  # Append this step's thought to the previous prompt
                else:
                    prompt_input = "\nobservation: "
                if not hdx: #hdx==0
                    prompt_input = private_instruction + prompt_input
                #print(ac,prompt_input,)
                messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[im]))
                messages.append(self.processor_wrapper.create_message_vllm(role="assistant",input_type="text",prompt=[ac],))
            
        prompt_input = ""
        if self.instruction_type == 'recipe':
            prompt_input = "\nthought: " + thought + "\nobservation: "
        else:
            prompt_input = "\nobservation: "
        if not self.history_num:
            prompt_input = private_instruction + prompt_input
        if verbos:
            print(f"=== PROMPT INPUT ===\n{prompt_input}\n====================")

        messages.append(self.processor_wrapper.create_message_vllm(role="user",input_type="image",prompt=[prompt_input],image=[image]))
        
        open_logprobs = False
        if verbos:
            print(prompts)
            open_logprobs = True
        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            top_p = 0.99,
            logprobs = open_logprobs,
            extra_body = {"skip_special_tokens":False, "top_k" : -1}
        )

        outputs = chat_completion.choices[0].message.content

        # Store the raw LLM output for display purposes
        self.last_llm_output = outputs
        self.last_thought = thought

        if self.history_num:
            new_history[-1] = (image,outputs,thought,self.history[-1][-1]+1)
            self.history = new_history
        if self.LLM_backbone in {"qwen2_vl"}:
            outputs = self.tokenizer(outputs)["input_ids"]

        actions =  self.action_tokenizer.decode(outputs)

        len_action = min(self.action_chunk_len,len(actions))
        self.actions = actions[:len_action]

        return self.actions.pop(0)

    # Logit bias dict to suppress action tokens (built once, reused)
    _vlm_suppress = None

    @classmethod
    def _get_vlm_suppress(cls):
        if cls._vlm_suppress is None:
            s = {str(tid): -100 for tid in range(151675, 151908)}
            s["677"] = -100    # [' wrapper
            s["151644"] = -100 # <|im_start|>
            cls._vlm_suppress = s
        return cls._vlm_suppress

    def query_vlm(self, observation, question, max_tokens=256, include_image=None):
        """Query the model in standard VLM mode (not trajectory mode).

        Stage III fine-tuning made the model output action tokens whenever it
        sees a Minecraft image. Text-only queries work much better for
        knowledge questions. By default, we auto-detect: if the question asks
        about the visual scene we include the image, otherwise text-only.

        Args:
            observation: numpy array of current game frame (may be ignored)
            question: natural language question
            max_tokens: max response length
            include_image: True/False to force, None to auto-detect
        """
        # Auto-detect whether the question needs visual context
        if include_image is None:
            visual_keywords = ['see', 'look', 'scene', 'describe', 'around',
                               'front', 'screen', 'image', 'frame', 'block',
                               'nearby', 'biome', 'mob', 'entity', 'wearing',
                               'holding', 'inventory']
            include_image = any(kw in question.lower() for kw in visual_keywords)

        if include_image:
            image = self.processor_wrapper.create_image_input(observation)
            messages = [
                self.processor_wrapper.create_message_vllm(
                    role="user", input_type="image",
                    prompt=[question], image=[image],
                )
            ]
        else:
            messages = [
                self.processor_wrapper.create_message_vllm(
                    role="user", input_type="text",
                    prompt=[question],
                )
            ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.99,
            logit_bias=self._get_vlm_suppress(),
            frequency_penalty=0.5,
            extra_body={
                "skip_special_tokens": True,
                "top_k": -1,
                "repetition_penalty": 1.3,
            },
        )

        return chat_completion.choices[0].message.content
