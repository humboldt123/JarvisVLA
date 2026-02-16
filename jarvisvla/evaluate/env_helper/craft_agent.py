import random
import math
import os 
import re
import json
import numpy as np
import time
import copy
from typing import (
    Sequence, List, Mapping, Dict, 
    Callable, Any, Tuple, Optional, Union
)
from minestudio.simulator.entry import MinecraftSim
from minestudio.models.shell.gui_agent import GUIWorker

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

class CraftWorker(GUIWorker):
    
    def __init__(
        self, 
        env: MinecraftSim,
        sample_ratio: float = 0.5,
        inventory_slot_range: Tuple[int, int] = (0, 36), 
        if_discrete = False,
        slow_act = True,
        recycle_craft_table = False,
        **kwargs, 
    )-> None:
        super().__init__(env=env,sample_ratio=sample_ratio,if_discrete=if_discrete,slow_act=slow_act,kwargs=kwargs)
        # print("Initializing worker...")
        self.inventory_slot_range = inventory_slot_range
        self.recycle_craft_table = recycle_craft_table
        self.crafting_slotpos = 'none'  

    # crafting
    def crafting(self, target: str, target_num: int=1, recipe_name:str=None):
        # recipe_name = "crafting_table.json"
        try:
            # is item/tag
            is_tag = False
            cur_path = os.path.abspath(os.path.dirname(__file__))
            root_path = cur_path[:cur_path.find('minestudio')]
            relative_path = os.path.join("assets/tag_items.json")
            tag_json_path = os.path.join(root_path, relative_path)
            with open(tag_json_path) as file:
                tag_info = json.load(file)
            for key in tag_info:
                if key[10:] == target:  # Strip "minecraft:" prefix (first 10 chars)
                    is_tag = True

            # open recipe one by one: only shapeless crafting like oak_planks        
            if is_tag:
                enough_material = False
                enough_material_target = 'none'
                item_list = tag_info['minecraft:'+target]

                for item in item_list:
                    subtarget = item[10:]
                
                    
                    recipe_path = recipe_name if recipe_name is not None else subtarget + '.json'
                    relative_path = os.path.join("assets/recipes", recipe_path)
                    recipe_json_path = os.path.join(root_path, relative_path)
                    with open(recipe_json_path) as file:
                        recipe_info = json.load(file)
                    need_table = self.crafting_type(recipe_info)

                    # find materials(shapeless) like oak_planks
                    ingredients = recipe_info.get('ingredients')
                    random.shuffle(ingredients)
                    items = dict()
                    items_type = dict()

                    # clculate the amount needed and store <item, quantity> in items
                    for i in range(len(ingredients)):
                        if ingredients[i].get('item'):
                            item = ingredients[i].get('item')[10:]
                            item_type = 'item'
                        else:
                            item = ingredients[i].get('tag')[10:]
                            item_type = 'tag'
                        items_type[item] = item_type
                        if items.get(item):
                            items[item] += 1
                        else:
                            items[item] = 1

                    if recipe_info.get('result').get('count'):
                        iter_num = math.ceil(target_num / int(recipe_info.get('result').get('count')))
                    else:
                        iter_num = target_num

                    enough_material_subtarget = True
                    for item, num_need in items.items():
                        labels = self.get_labels()
                        inventory_id = self.find_in_inventory(labels, item, items_type[item])
                        if not inventory_id:
                            enough_material_subtarget = False
                            break
                        inventory_num = labels.get(inventory_id).get('quantity')
                        if num_need * iter_num > inventory_num:
                            enough_material_subtarget = False
                            break
                    if enough_material_subtarget:
                        enough_material = True
                        enough_material_target = subtarget

                if enough_material:
                    target = enough_material_target
                else:
                    self._assert(0, f"not enough materials for {target}")

            # if inventory is open by accident, close inventory
            if self.slow_act:
                self._null_action(1)
            if self.info['isGuiOpen']:
                self._press_inventory_button()
                
            cur_path = os.path.abspath(os.path.dirname(__file__))
            root_path = cur_path[:cur_path.find('minestudio')]
            recipe_path = recipe_name if recipe_name is not None else target + '.json'
            relative_path = os.path.join("assets/recipes", recipe_path)
            recipe_json_path = os.path.join(root_path, relative_path)
            with open(recipe_json_path) as file:
                recipe_info = json.load(file)
            need_table = CraftWorker.crafting_type(recipe_info)

            if need_table:
                self.open_crating_table_wo_recipe()
            else:
                self.open_inventory_wo_recipe()               
            # crafting
            if recipe_info.get('result').get('count'):
                iter_num = math.ceil(target_num / int(recipe_info.get('result').get('count')))
            else:
                iter_num = target_num
            
            self.crafting_once(target, iter_num, recipe_info, target_num)
            # close inventory
            self._press_inventory_button()
            if need_table and self.recycle_craft_table:
                self.return_crafting_table()
            self.current_gui_type = None
            self.crafting_slotpos = 'none'  

        except AssertionError as e:
            return False, str(e) 
        
        return True, None

    # open inventory    
    def open_inventory_wo_recipe(self):
        self._press_inventory_button()
        # update slot pos
        self.current_gui_type = 'inventory_wo_recipe'
        self.crafting_slotpos = self.slot_pos_inventory_wo_recipe
        self.roam_camera()
        
    # before opening crafting_table
    def pre_open_tabel(self, attack_num=20):
        action = self.env.noop_action()
        self.obs, _, _, _, self.info = self._step(action)
        height_1 = self.info['location_stats']['ypos']

        action['jump'] = 1
        self.obs, _, _, _, self.info = self._step(action)
        height_2 = self.info['location_stats']['ypos']

        self._null_action(1)
        if height_2 - height_1 > 0.419:
            pass
        else:
            '''euip pickaxe'''
            self.obs, _, _, _, self.info = self._step(action)
            height = self.info['location_stats']['ypos']
            if height < 50:
                # find pickaxe
                labels = self.get_labels()
                inventory_id_diamond = self.find_in_inventory(labels, 'diamond_pickaxe', 'item')
                inventory_id_iron = self.find_in_inventory(labels, 'iron_pickaxe', 'item')
                inventory_id_stone = self.find_in_inventory(labels, 'stone_pickaxe', 'item')
                inventory_id_wooden = self.find_in_inventory(labels, 'wooden_pickaxe', 'item')

                if inventory_id_wooden:
                    inventory_id = inventory_id_wooden
                if inventory_id_stone:
                    inventory_id = inventory_id_stone
                if inventory_id_iron:
                    inventory_id = inventory_id_iron
                if inventory_id_diamond:
                    inventory_id = inventory_id_diamond
                
                if inventory_id != 'inventory_0':
                    self.open_inventory_wo_recipe()
                    
                    '''clear inventory 0'''
                    labels=self.get_labels()
                    item_in_slot_0 = labels['inventory_0']['type']
                    if labels['inventory_0']['type'] != 'none':
                        for i in range(9):
                            del labels["resource_"+str(i)]
                        inventory_id_none = self.find_in_inventory(labels, 'none')
                        self.pull_item_all(self.crafting_slotpos, 'inventory_0', inventory_id_none)
                    
                    self.pull_item(self.crafting_slotpos, inventory_id, 'invelabelsntory_0',item_in_slot_0, 1)
                    self._press_inventory_button()
                    self.current_gui_type = None
                    self.crafting_slotpos = 'none'
                    self._call_func('hotbar.1')

            action = self.env.noop_action()
            for i in range(2):
                action['camera'] = np.array([-88, 0])
                self.obs, _, _, _, self.info = self._step(action)

            action['camera'] = np.array([22, 0])
            self.obs, _, _, _, self.info = self._step(action)   

            for i in range(5):
                action['camera'] = np.array([0, 60])
                self.obs, _, _, _, self.info = self._step(action)
                self._attack_continue(attack_num)

    # open crafting_table
    def open_crating_table_wo_recipe(self):
        self.pre_open_tabel()
        if self.slow_act:
            self._null_action(1)
        if self.info['isGuiOpen']:
            self._press_inventory_button()      
        self.open_inventory_wo_recipe()
        for i in range(10):
            self._null_action(1)
            if self.info['isGuiOpen']:
                self.place = "inventory"
                break
        labels=self.get_labels()
        inventory_id = self.find_in_inventory(labels, 'crafting_table')
        self._assert(inventory_id, f"no crafting_table {labels}")

        if inventory_id != 'inventory_0':
            labels=self.get_labels()
            if labels['inventory_0']['type'] != 'none':
                for i in range(9):
                    del labels["resource_"+str(i)]
                inventory_id_none = self.find_in_inventory(labels, 'none')
                self.pull_item_all(self.crafting_slotpos, 'inventory_0', inventory_id_none)
            self.pull_item(self.crafting_slotpos, inventory_id, 'inventory_0','crafting_table', 1)
        
        # Close inventory
        self._press_inventory_button()
        self.current_gui_type = None
        self.crafting_slotpos = 'none'
        for i in range(10):
            self._null_action(1)
            if not self.info['isGuiOpen']:
                break
        self._call_func('hotbar.1') 
        # Place crafting table on the ground
        self._place_down()
        for i in range(11):
            self._call_func('use') # Open crafting table
            if i>0 and i%10==0:
                time.sleep(0.5)
            if self.info['isGuiOpen']:
                break
        self._assert(self.info['isGuiOpen'],"the isGuiOpen is not True")
        self._null_action(2)

        forget_frames,forget_infos,forget_actions = self.forget(num=0)
        self._reset_cursor() # Reset cursor position
        self.current_gui_type = 'crating_table_wo_recipe'
        self.crafting_slotpos = self.slot_pos_table_wo_recipe
        return forget_frames,forget_infos,forget_actions        

    def random_move_or_stay(self,random_p=[0.5,0.25]):
        if np.random.uniform(0, 1) > random_p[0]:
            num_random = random.randint(2, 4)
            if random.uniform(0, 1) > random_p[1]:
                for i in range(num_random):
                    self.move_once(0, 0)
            else:
                for i in range(num_random):
                    d1 =  random.uniform(-5, 5)
                    d2 =  random.uniform(-5, 5)
                    self.move_once(d1, d2)
        else:
            pass
    
    def move_to_slot(self, SLOT_POS: Dict, slot: str):
        self._assert(slot in SLOT_POS, f'Error: slot: {slot}')
        x, y = SLOT_POS[slot]
        self.move_to_pos(x, y)
    
    # pull
    # select item_from, select item_to
    def pull_item_all(self, 
        SLOT_POS: Dict, 
        item_from: str, 
        item_to: str
    ) -> None:
        self.move_to_slot(SLOT_POS, item_from)
        if self.slow_act:
            self._null_action(1)
        self._select_item()
        if self.slow_act:
            self._null_action(1)
        self.move_to_slot(SLOT_POS, item_to) 
        if self.slow_act:
            self._null_action(1)
        self._select_item()
        if self.slow_act:
            self._null_action(1)
        self.random_move_or_stay()
    
    # select item_from, use n item_to    
    def pull_item(self, 
        SLOT_POS: Dict, 
        item_from: str, 
        item_to: str,
        item: str,
        target_number: int
    ) -> None:
        raw_item_name = copy.deepcopy(item)
        if 'resource' in item_to:
            item = self.info['inventory'][int(item_from.split('_')[-1])]
            self.resource_record[item_to] = item
            slot_id = "_input_slot." + item_to[8:]
            if self.current_gui_type is None:
                pass
            elif self.current_gui_type == 'crating_table_wo_recipe':
                slot_id = "table" + slot_id
            elif self.current_gui_type == 'inventory_wo_recipe':
                slot_id = "inventory" + slot_id
                
        self.move_to_slot(SLOT_POS, item_from)
        if self.slow_act:
            self._null_action(1)
        
        self._select_item()
        self.gui["cursor"] = raw_item_name
        self.move_to_slot(SLOT_POS, item_to)
        if self.slow_act:
            self._null_action(1)
        del self.gui["cursor"]
        self.gui[slot_id] = raw_item_name
        for i in range(target_number):
            self._use_item()
            if self.slow_act:
                self._null_action(1)
        
    # use n item_to 
    def pull_item_continue(self, 
        SLOT_POS: Dict, 
        item_to: str,
        item: str,
        target_number: int,
    ) -> None:
        if 'resource' in item_to:
            self.resource_record[item_to] = item
            slot_id = "_input_slot." + item_to[8:]
            if self.current_gui_type is None:
                pass
            elif self.current_gui_type == 'crating_table_wo_recipe':
                slot_id = "table" + slot_id
            elif self.current_gui_type == 'inventory_wo_recipe':
                slot_id = "inventory" + slot_id
        
        self.gui["cursor"] = item
        self.move_to_slot(SLOT_POS, item_to)
        del self.gui["cursor"]
        self.gui[slot_id] = item
        if self.slow_act:
            self._null_action(1)
        for i in range(target_number):
            self._use_item()
            if self.slow_act:
                self._null_action(1)
        
        self.random_move_or_stay()
    
    # select item_to 
    def pull_item_return(self, 
        SLOT_POS: Dict, 
        item_to: str,
        item: str,
    ) -> None: 
        self.gui["cursor"] = item
        self.move_to_slot(SLOT_POS, item_to)
        if self.slow_act:
            self._null_action(1)
        del self.gui["cursor"]
        self._select_item()
        if self.slow_act:
            self._null_action(1)
        self.random_move_or_stay()
        
    # use n item_frwm, select item_to
    def pull_item_result(self, 
        SLOT_POS: Dict, 
        item_from: str,
        item_to: str,
        target_number: int,
        item: str,
    ) -> None: 
        
        self.move_to_slot(SLOT_POS, item_from)
        for i in range(target_number):
            self._use_item()
            if self.slow_act:
                self._null_action(1)
        self.gui = {}
        self.gui["cursor"] = item
        self.move_to_slot(SLOT_POS, item_to)
        del self.gui["cursor"]
        self._select_item()
        if self.slow_act:
            self._null_action(1)
        self.random_move_or_stay()

    # get all labels
    def get_labels(self, noop=True):
        if noop:
            if self.slow_act:
                self._null_action(1)
        result = {}
        # generate resource recording item labels
        for i in range(9):
            slot = f'resource_{i}'
            item = self.resource_record[slot]
            result[slot] = item
        
        # generate inventory item labels
        for slot, item in self.info['inventory'].items():
            result[f'inventory_{slot}'] = item
        
        return result
    
    # return crafting table
    def return_crafting_table(self):
        self._look_down()
        labels = self.get_labels()
        table_info = self.find_in_inventory(labels, 'crafting_table')
        tabel_exist = 0
        if table_info:
            tabel_exist = 1
            tabel_num = labels.get(table_info).get('quantity')

        done = 0
        for i in range(4):
            for i in range(10):
                self._attack_continue(8)
                labels = self.get_labels(noop=False)
                if tabel_exist:
                    table_info = self.find_in_inventory(labels, 'crafting_table')
                    tabel_num_2 = labels.get(table_info).get('quantity')
                    if tabel_num_2 != tabel_num:
                        done = 1
                        break
                else:
                    table_info = self.find_in_inventory(labels, 'crafting_table')
                    if table_info:
                        done = 1
                        break
            self._call_func('forward')
        self._assert(done, f'return crafting_table unsuccessfully')    

    # judge crafting_table / inventory
    @staticmethod
    def crafting_type(target_data: Dict):
        if 'pattern' in target_data:
            pattern = target_data.get('pattern')
            col_len = len(pattern)
            row_len = len(pattern[0])
            if col_len <= 2 and row_len <= 2:
                return False
            else:
                return True
        else:
            ingredients = target_data.get('ingredients')
            item_num = len(ingredients)
            if item_num <= 4:
                return False
            else:
                return True
    
    # search item in agent's inventory 
    def find_in_inventory(self, labels: Dict, item: str, item_type: str='item', path=None):

        if path == None:
            path = []
        for key, value in labels.items():
            current_path = path + [key]
            if item_type == "item":
                if re.match(item, str(value)):
                    return current_path
                elif isinstance(value, dict):
                    result = self.find_in_inventory(value, item, item_type, current_path)
                    if result is not None:
                        return result[0]
            elif item_type == "tag":
                # tag info
                cur_path = os.path.abspath(os.path.dirname(__file__))
                root_path = cur_path[:cur_path.find('minestudio')]
                relative_path = os.path.join("assets/tag_items.json")
                tag_json_path = os.path.join(root_path, relative_path)
                with open(tag_json_path) as file:
                    tag_info = json.load(file)

                item_list = tag_info['minecraft:'+item]
                for i in range(len(item_list)):
                    if re.match(item_list[i][10:], str(value)):
                        return current_path
                    elif isinstance(value, dict):
                        result = self.find_in_inventory(value, item, item_type, current_path)
                        if result is not None:
                            return result[0]
        return None

    # crafting once 
    def crafting_once(self, target: str, iter_num: int, recipe_info: Dict, target_num:int):
        # shaped crafting
        
        if "pattern" in recipe_info:
            self.crafting_shaped(target, iter_num, recipe_info)
        # shapless crafting
        else:
            self.crafting_shapeless(target, iter_num, recipe_info)
        
        # get result
        # Do not put the result in resource
        labels = self.get_labels()
        for i in range(9):
            del labels["resource_"+str(i)]
        
        # Check if the target item already exists in inventory
        result_inventory_id_1 = self.find_in_inventory(labels, target)
        
        if result_inventory_id_1:
            item_num = labels.get(result_inventory_id_1).get('quantity')
            if item_num + target_num < 60:
                self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_1, iter_num, target)
                labels_after = self.get_labels()
                item_num_after = labels_after.get(result_inventory_id_1).get('quantity')
                if item_num == item_num_after:
                    result_inventory_id_2 = self.find_in_inventory(labels, 'none')
                    self._assert(result_inventory_id_2, f"no space to place result")
                    self.pull_item_return(self.crafting_slotpos, result_inventory_id_2, target)
                    self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason 111111")
            else:

                result_inventory_id_2 = self.find_in_inventory(labels, 'none')
                self._assert(result_inventory_id_2, f"no space to place result")
                self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_2, iter_num, target)
                self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason 22222")
        else:
            # Find an empty slot
            result_inventory_id_2 = self.find_in_inventory(labels, 'none')
            
            self._assert(result_inventory_id_2, f"no space to place result")
            
            self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_2, iter_num, target)

            self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason 33333")
        
        # clear resource          
        self.resource_record =  {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)}
        
    # shaped crafting
    def crafting_shaped(self, target:str, iter_num:int, recipe_info: Dict,shuffle_p=0): # 1 - shuffle_p: probability of shuffling
        slot_pos = self.crafting_slotpos
        labels = self.get_labels()
        pattern = recipe_info.get('pattern')
        items = recipe_info.get('key')
        items = random_dic(items) # Randomize order
        
        # place each item in order
        if self.current_gui_type == 'crating_table_wo_recipe': # Should be done on open, but doesn't work there
            self._null_action()
            self.roam_camera()
            self.forget(num=0)
        for k, v in items.items():
            signal = k
            if v.get('item'):
                item = v.get('item')[10:]
                item_type= 'item'
            else:
                item = v.get('tag')[10:]
                item_type= 'tag'
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, item, item_type)
            self._assert(inventory_id, f"not enough {item}")
            inventory_num = labels.get(inventory_id).get('quantity')

            # clculate the amount needed
            num_need = 0
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        num_need += 1
            num_need = num_need * iter_num
            self._assert(num_need <= inventory_num, f"not enough {item}，wants {num_need}, has {inventory_num}")

            # place
            resource_idx = 0
            first_pull = 1
            if 'table' in self.current_gui_type:
                type = 3
            else:
                type = 2
            resource_ids = []
            for i in range(len(pattern)):
                resource_idx = i * type
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        resource_ids.append([i,j,resource_idx])
                        """ 
                        if first_pull:
                            self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx), iter_num)
                            first_pull = 0
                        else:
                            self.pull_item_continue(slot_pos, 'resource_' + str(resource_idx), item, iter_num)
                        """
                    resource_idx += 1
            if np.random.uniform(0, 1) > shuffle_p:
                random.shuffle(resource_ids)

            for i,j,resource_idx in resource_ids:
                if first_pull:
                    self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx),item, iter_num)
                    first_pull = 0
                else:
                    self.pull_item_continue(slot_pos, 'resource_' + str(resource_idx), item, iter_num)
            
            # return the remaining items
            if num_need < inventory_num:
                self.pull_item_return(slot_pos, inventory_id,target)
        
        slot_id = ""
        if self.current_gui_type is None:
            pass
        elif self.current_gui_type == 'crating_table_wo_recipe':
            slot_id = "table_output_slot"
        elif self.current_gui_type == 'inventory_wo_recipe':
            slot_id = "inventory_output_slot"
        self.gui[slot_id] = target
    
    # shapeless crafting
    def crafting_shapeless(self, target:str, iter_num:int, recipe_info: Dict):   
        slot_pos = self.crafting_slotpos 
        labels = self.get_labels()
        ingredients = recipe_info.get('ingredients')
        random.shuffle(ingredients)
        items = dict()
        items_type = dict()

        # clculate the amount needed and store <item, quantity> in items
        for i in range(len(ingredients)):
            if ingredients[i].get('item'):
                item = ingredients[i].get('item')[10:]
                item_type = 'item'
            else:
                item = ingredients[i].get('tag')[10:]
                item_type = 'tag'
            items_type[item] = item_type
            if items.get(item):
                items[item] += 1
            else:
                items[item] = 1
        
        # place each item in order
        resource_idx = 0
        for item, num_need in items.items():
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, item, items_type[item])
            self._assert(inventory_id, f"not enough {item}")
            inventory_num = labels.get(inventory_id).get('quantity')
            self._assert(num_need * iter_num <= inventory_num, f"not enough {item}，wants {num_need * iter_num}, has {inventory_num}")

            # place 
            num_need -= 1
            self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx),item, iter_num)

            resource_idx += 1
            if num_need != 0:
                for i in range(num_need):
                    self.pull_item_continue(slot_pos, 'resource_' + str(resource_idx), item, iter_num)
                    resource_idx += 1
            
            # return the remaining items
            num_need = (num_need + 1) * iter_num
            if num_need < inventory_num:
                self.pull_item_return(slot_pos, inventory_id, target)
        
        slot_id = ""
        if self.current_gui_type is None:
            pass
        elif self.current_gui_type == 'crating_table_wo_recipe':
            slot_id = "table_output_slot"
        elif self.current_gui_type == 'inventory_wo_recipe':
            slot_id = "inventory_output_slot"
        self.gui[slot_id] = target

if __name__ == '__main__':

    # cur_path = os.path.abspath(os.path.dirname(__file__))
    # root_path = cur_path[:cur_path.find('jarvis')]
    # relative_path = os.path.join("jarvis/assets/tag_items.json")
    # tag_json_path = os.path.join(root_path, relative_path)
    # with open(tag_json_path) as file:
    #     tag_info = json.load(file)
    # for key in tag_info:
    #     print(key[10:])
  
    import numpy as np
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        SpeedTestCallback, 
        RecordCallback, 
        RewardsCallback, 
        TaskCallback,
        FastResetCallback,
        InitInventoryCallback
    )
    sim = MinecraftSim(
        action_type="env",
        callbacks=[
            SpeedTestCallback(50), 
            TaskCallback([
                {'name': 'craft', 'text': 'craft crafting_table'}, 
            ]),
            RewardsCallback([{
                'event': 'craft_item', 
                'objects': ['crafting_table'], 
                'reward': 1.0, 
                'identity': 'craft crafting_table', 
                'max_reward_times': 1, 
            }]),
            RecordCallback(record_path="output", fps=30,record_actions=True,record_infos=True,record_origin_observation=True),
            FastResetCallback(
                biomes=['plains'],
                random_tp_range=1000,
            ), 
            
            InitInventoryCallback([
                {"slot": 0,
                "type": "diamond_sword",
                "quantity":1,},
            ],inventory_distraction_level="random")
        ]
    )
    obs, info = sim.reset()
    action = sim.noop_action()
    obs, reward, terminated, truncated, info = sim.step(action)
    
    worker = CraftWorker(sim)
    done, info = worker.crafting('wooden_pickaxe', 1)
    sim.close()
    #done, info = worker.crafting('wooden_pickaxe', 1)
    
    #worker = Worker('test')
    #done, info = worker.crafting('wooden_pickaxe', 2)
    #print(done, info)
    #write_video('crafting.mp4', worker.outframes)