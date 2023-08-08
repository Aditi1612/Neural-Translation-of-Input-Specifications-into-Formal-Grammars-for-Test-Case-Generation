import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
import jsonlines
from tqdm.notebook import tqdm

model_params = {
    "MODEL": "Salesforce/codet5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 64,  # training batch size
    "VALID_BATCH_SIZE": 64,  # validation batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "VAL_EPOCHS": 2,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

def load_data(path,tokenizer):
    df = pd.DataFrame()
    
    # df['source'] = []
    problems = []
    sources = []
    
    with jsonlines.open(f'data/{path}.jsonl') as f:
        for problem in f:
            if problem['description']['spec'] == '':
                continue
            if len(sources) >= 3000: break
            source = problem['description']['spec'] + tokenizer.eos_token
            sources.append(source)
            # df['source'].append(problem['description']['spec'] + tokenizer.eos_token)
            problems.append(problem)
            # df['source']  = sources
            
    
    # with open(f'data/{path}.jsonl', encoding='utf-8') as f:
    #     for line in f:
    #         sources=[]
    #         targets=[]
    #         obj = json.loads(line)
    #         if len(obj['solutions']) == 0 or len(obj['incorrect_solutions']) == 0: continue
    #         # input_rule = obj['input_rule']
    #         for source in obj['incorrect_solutions']:
    #             # source += tokenizer.sep_token + input_rule
    #             # if len(source) > model_params['MAX_SOURCE_TEXT_LENGTH']: continue
    #             sources.append(source)
    #             targets.append(tokenizer.eos_token)
    #         df=pd.DataFrame()
            
    #         if len(sources) == 0: continue
            
    #         test_data.append(obj)
    #         df['source']=sources
    #         df['target']=targets
    #         df_list.append(df)
        
    return sources, problems

class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        # source_text = " ".join(source_text.split())
        # target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        # print(source['attention_mask'])
        # exit(-1)
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
    
def inference(model,data,tokenizer,device):
    """
    Inference function for the model
    """
    model.eval()
    
    # cleaning data so as to ensure data is in string type
    # source_text = " ".join(data.split())

    source = tokenizer.batch_encode_plus(
        [data],
        max_length=512,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    source_ids = source["input_ids"].to(device, dtype = torch.long)
    source_mask = source["attention_mask"].to(device, dtype = torch.long)
    print(source["attention_mask"])
    print(source_mask)
    exit()

    generated_ids = model.generate(
    input_ids = source_ids,
    attention_mask = source_mask, 
    max_length=150, 
    # num_beams=20,
    repetition_penalty=2.5, 
    length_penalty=1.0, 
    early_stopping=True,
    # num_return_sequences=10
    )
    
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds

if __name__ == '__main__':
    import sys
    
    device = torch.device("cuda:0") 
    '''
    dataset_name = sys.argv[1]
    model_file_name = sys.argv[2]
    model_file_idx = sys.argv[3]
    '''
    dataset_name = "train_grammar"
    model_file_name = "origin_changed_learning_rate_1e-4"
    # model_file_idx = "19"
    
    
    # count = 0
    # blink_list = [''] * 10

    # for idx in [0, 1]: # range(1, 6):
    for idx in range (20):
        model_file_idx = idx
        model_name = f'{model_file_name}{model_file_idx}'
        print(model_name)
        tokenizer = RobertaTokenizer.from_pretrained(f'outputs/{model_name}')
        model = T5ForConditionalGeneration.from_pretrained(f'outputs/{model_name}')
        sep_token = tokenizer.sep_token
        model.to(device)

        df, problems = load_data(dataset_name, tokenizer)
        with open(f'outputs/{model_name}/result{model_file_idx}_greedy.jsonl', 'w',  encoding='utf-8') as write_file:
                write_file.write('')
        # io = []
        
        for dataset_idx, (data, problem) in enumerate(zip(df, problems), 1):
            if dataset_idx % 10 == 0:
                print(dataset_idx//10)
            
            
            predictions=inference(model,data,tokenizer,device)
            problem['spec'] = {}
            problem['spec']['generated'] = predictions
            
            with open(f'outputs/{model_name}/result{model_file_idx}_greedy.jsonl', 'a',  encoding='utf-8') as write_file:
                write_file.write(json.dumps(problem, ensure_ascii=False) + '\n')
    '''
    for dataset_idx, (df, problem) in enumerate(zip(df_list, dataset), 1):
        if dataset_idx % 100 == 0:
            print(int(dataset_idx/100))

        # problem = {}
        # for key in line:
        incorrect_solutions = problem['incorrect_solutions']
        
        # generated_test_case = []
        problem['incorrect_solutions'] = []
        problem['generated_tests'] = []
        generated_tests = {'input' : [], 'output' : []}
        for data, solution in zip(df['source'], incorrect_solutions):
            
            predictions=inference(model,data,tokenizer,device)

            # generated_test_case.update(predictions)
                # generated_test_case['output'].append('')

            generated_tests['input'] = predictions
            generated_tests['output'] = []
            
            problem['incorrect_solutions'].append(solution)
            problem['generated_tests'].append(generated_tests)
        # output_dict = {'problem_name': name, 'Solutions' : data, 'generated_test_case' : generated_test_case}
        '''

        # io.append(line)
    # io
    
    # print(count)
    # count = 0
        
