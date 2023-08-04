import os
import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
# import logger


# Importing the T5 modules from huggingface/transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import json
from tqdm.notebook import tqdm
from torch import cuda



device = 'cuda:0'
#device = 'cpu'
# print(f'tensorboard --logdir=./logs/{save_model_name}')

model_params = {
    "MODEL": "Salesforce/codet5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    # "TRAIN_EPOCHS": 10,  # number of training epochs\
    "TRAIN_EPOCHS" : 50,
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-7,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
    "NUM_OF_BEAM_SAMPLE": 10,
}

save_model_name = f'origin_changed_learning_rate_1e-7_50_epoch'
log_dir = f'./logs/{save_model_name}/'
writer = SummaryWriter(log_dir)

# tokenizer = RobertaTokenizer.from_pretrained(model_params["MODEL"])
# # print_log = logger.logger(log_file_path).# print_log

'''
def # print_log(text :str):
    current_date = date.today()
    current_time = time.strftime("%H:%M:%S", time.localtime())

    with open(log_file_path, 'a', encoding="utf-8") as log_file:
        log_file.write(f'{current_date} {current_time} : {text}' + '\n')
'''

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
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def load_data(path,tokenizer):
    '''
    with open(f'data/{path}.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            obj=json.loads(line)
            # source=obj['description']+tokenizer.sep_token+obj['solutions']
            source = obj['solutions']
            
            for t in obj['test_cases']:
                sources.append(source)
                targets.append(t+tokenizer.eos_token)
            for t in obj['private_tests']:
                sources.append(source)
                targets.append(t+tokenizer.eos_token)
            if idx>50000:
                break
    '''
    
    sources=[]
    targets=[]
    indexs = []
    names = []
    
    # print(tokenizer.sep_token)
    # print(tokenizer.eos_token)
    with jsonlines.open(f'data/{path}.jsonl') as f:
        for idx, obj in enumerate(f):
           
            source = obj['description']['spec']
            if source == '':
                print(obj)
                continue
                # exit()
            indexs.append(obj['name']['index'])
            names.append(obj['name']['name'])
            sources.append(source)
            # target = ','.join(obj['spec']['grammer'])
            target1 = ' / '.join(obj['spec']['grammer'])
            target2 = ' / '.join(obj['spec']['constraints']) 
            target1 = target1.replace("<S>", "<R>").replace("<s>", "<p>")
            target = target1 + ' // ' + target2
            targets.append(target + tokenizer.eos_token)
            
            # if len(source) > model_params['MAX_SOURCE_TEXT_LENGTH']: continue
            
            # for test_case in obj['input']
            
            '''
            for test_case in obj['test_case_inputs']:
                if len(test_case) > model_params['MAX_TARGET_TEXT_LENGTH'] or len(test_case) < 1: continue
                sources.append(source)
                targets.append(test_case + tokenizer.eos_token)
            
            for test_case in obj['public_tests']['test_case_inputs']:
                if len(test_case) > model_params['MAX_TARGET_TEXT_LENGTH'] or len(test_case) < 1: continue
                sources.append(source)
                targets.append(test_case + tokenizer.eos_token)
            
            for test_case in obj['private_tests']['input']:
                if len(test_case) > model_params['MAX_TARGET_TEXT_LENGTH'] or len(test_case) < 1: continue
                sources.append(source)
                targets.append(test_case + tokenizer.eos_token)
            '''
            # targets.append(obj['public_tests']['input'][0]+tokenizer.eos_token)

    df=pd.DataFrame()
    df['index'] = indexs
    df['name'] = names
    df['source']=sources
    df['target']=targets
    
    # df = df.sample(frac=0.025, random_state=model_params['SEED'])

    print(len(df['source']))
    # print_log(f'{len(df["source"])}')

    return df

def train(epoch, tokenizer, model, device, loader, optimizer):
    print(epoch, "th train start")
    # print_log(f"{epoch} th train start")
    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    total_loss = 0
    # curr_loss = 99
    for idx, data in enumerate(loader, 1):
        # print(i)
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        # print(loss.shape)
        # print(loss)

        # writer.add_scalar("Loss/train", loss, epoch)
        
        total_loss += loss.item()
        if idx % 50 == 0:
            aver_loss = total_loss / idx
            # total_loss = 0
            print('{:7}: {}'.format(int(idx/50), aver_loss))
            # print_log('{:7}: {}'.format(int(i/1000), aver_loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # exit(-1)
    print(epoch, "th train end")
    print('{:7}: {}'.format(idx, total_loss / idx))
    # print_log(f"{epoch} th train end")
    writer.add_scalar('Mean_Loss/train', total_loss/idx, epoch)
    return loss

def validate(epoch, tokenizer, model, device, loader):
    
    print("validate start")
    # print_log("validate start")
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    actuals = []
    sources = []
    
    skip_special_tokens = True
    
    with torch.no_grad():
        total_loss = 0
        for idx, data in enumerate(loader, 1): 
            # print(idx)
            res = []
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=model_params['MAX_TARGET_TEXT_LENGTH'], 
                num_beams=10,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                num_return_sequences=model_params['NUM_OF_BEAM_SAMPLE']
                )
            loss = model(input_ids=ids, labels=y).loss
            total_loss += loss.item()

            # if idx == 1: print(type(generated_ids))

            # if idx % 1000 == 0:
            #     print("{:10} prediction loss: {}".format(int(idx/1000), loss/1000))
            #     loss = 0

            # print(len(generated_ids))
            preds  = [tokenizer.decode(g, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True) for t in y]
            source = [tokenizer.decode(i, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=True) for i in ids]
            sources.extend(source)
            # predictions.extend(preds)
            actuals.extend(target)
            # print(preds)
            num_of_problem = len(preds) // model_params['NUM_OF_BEAM_SAMPLE']
            result = []
            for i in range(num_of_problem):
                result.append(preds[:model_params['NUM_OF_BEAM_SAMPLE']])
                del preds[:model_params['NUM_OF_BEAM_SAMPLE']]
            # print(result)
            
            predictions.extend(result)
            
            # print(len(target))
            # print(len(source))
            # print(len(preds))
            """sources.extend(source)
            res.extend(preds)
            actuals.extend(target)
            
            # exit()
            """
            # predictions = result
    print("validate end")
    print(total_loss / idx)
    # print_log("validate end")
    writer.add_scalar('Mean_Loss/valid', total_loss/idx, epoch)
    return predictions, actuals, sources, loss

def T5Trainer(
      output_dir="./outputs/",
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = RobertaTokenizer.from_pretrained(model_params["MODEL"])
    
    '''
    added_toks = {}
    added_toks['bos_token'] = "<\\b>"
    added_toks['cls_token'] = "<\\c>"
    added_toks['sep_token'] = "<\\p>"
    added_toks['eos_token'] = "<\\e>"
    # num_added_toks = {"bos_token": "<bos>", "cls_token": "<cls>", "sep_token": "<s>", "mask_token": "<mask>"}
    # special_tokens_dict = {'additional_special_tokens': new_special_tokens + tokenizer.all_special_tokens}
    tokenizer.add_special_tokens(added_toks)
    
    # tokenizer.sep_token = "<sep>"
    # tokenizer.bos_token = "<bos>"
    # tokenizer.eos_token = "<eos>"
    # tokenizer.cls_token = "<cls>"
    '''
    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)


    # Importing the raw dataset
    # dataframe = dataframe[[source_text, target_text]]
   
    # train_dataset = load_data('train_grammer1',tokenizer)
    train_dataset = load_data('train_grammar', tokenizer)
    
    valid_dataset = load_data('test_grammar', tokenizer)
    source_text='source'
    target_text='target'
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    # train_dataset = dataframe.sample(n=700, random_state=model_params["SEED"])
    # val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=False)
    # train_dataset = train_dataset.reset_index(drop=False)
    # print(valid_dataset)
  

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        valid_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )
    print('starting training')
    # print_log('starting training')
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        loss = train(epoch, tokenizer, model, device, training_loader, optimizer)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.flush()

        path = os.path.join(output_dir, f"{save_model_name}{epoch}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        
        # if epoch in [1, 2, 4, 6] : continue
        predictions, targets, sources, loss = validate(epoch, tokenizer, model, device, val_loader)
        writer.add_scalar("Loss/valid", loss, epoch)
        writer.flush()
        # final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": target,"Source Text":sources})
        # print(len(sources))
        # print(len(targets))
        # print(len(predictions))
      
        with open(f"outputs/{save_model_name}_res{epoch}.jsonl", 'w') as f:
            for predict, target, source, index, name in zip(predictions, targets, sources, valid_dataset['index'], valid_dataset['name']):

                # target = target.replace("<R>", "<S>").replace("<p>", "<s>")
                
                # predict = predict.replace(tokenizer.pad_token, '')
                # target = target.replace(tokenizer.pad_token, '')
                # source = source.replace(tokenizer.pad_token, '')
                # print(predict)
       
                f.write(json.dumps({'index': index, 'name': name, 'source': source, 'target': target, 'generated': predict}, ensure_ascii=False) + '\n')
        # final_df.to_csv(os.path.join(f'{output_dir}{save_model_name}{epoch}/', f"predictions_train{epoch}.csv"))
    # Saving the model after training
        # predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        # final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        # final_df.to_csv(os.path.join(output_dir, f"predictions_valid{epoch}.csv"))
    writer.close()
if __name__ == '__main__':
    # # print_log = logger.logger('log',log_file).# print_log
    T5Trainer()
