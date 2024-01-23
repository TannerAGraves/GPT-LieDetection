import os
from openai import OpenAI
import pandas as pd
import json

key = os.environ.get("OPENAI_API_KEY")
client=OpenAI(api_key=key)

import json

def sanitize_to_utf8(input_str):
    return input_str.encode('utf-8', 'replace').decode('utf-8')

def process_jsonl_file(input_jsonl_path, output_jsonl_path):

    with open(input_jsonl_path, 'r', encoding='utf-8') as input_file, \
         open(output_jsonl_path, 'w', encoding='utf-8') as output_file:

        for line in input_file:
            json_obj = json.loads(line)
            sanitized_json_obj = {key: sanitize_to_utf8(value) if isinstance(value, str) else value
                                  for key, value in json_obj.items()}
            output_file.write(json.dumps(sanitized_json_obj) + '\n')


input_jsonl_path = 'train_data_full.json'
output_jsonl_path = 'output.jsonl'

process_jsonl_file(input_jsonl_path, output_jsonl_path)

sys_prompt = "You are an expert capable of discerning truthful from deceptive opinions based on speech patterns."

def gen_finetune(input, output, test=False):
    with open(input, 'r', encoding='utf-8') as data_in, \
        open(output, 'w') as gpt_out:
        for i, line in enumerate(data_in):
            user_prompt = json.loads(line)['text']
            sys_reply = "True" if json.loads(line)["label"] == 1 else "False"
            if not test:
                payload = {"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", 
                                    "content": user_prompt}, {"role": "assistant", "content": sys_reply}]}
            else:# exclude response from test set
                payload = {"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", 
                                                                        "content": user_prompt}]}
            gpt_out.write(json.dumps(payload) + '\n')

# 3 fold CV
gen_finetune('train_data_CV1.json', 'trainCV1_gpt.jsonl')
gen_finetune('val_data_CV1.json', 'valCV1_gpt.jsonl')
gen_finetune('train_data_CV2.json', 'trainCV2_gpt.jsonl')
gen_finetune('val_data_CV2.json', 'valCV2_gpt.jsonl')
gen_finetune('train_data_CV3.json', 'trainCV3_gpt.jsonl')
gen_finetune('val_data_CV3.json', 'valCV3_gpt.jsonl')

gen_finetune('train_data_300.json', 'train300_gpt.jsonl')
gen_finetune('val_data_300.json', 'val300_gpt.jsonl')

gen_finetune('train_data_full.json', 'trainfull_gpt.jsonl')
gen_finetune('val_data_full.json', 'valfull_gpt.jsonl')

gen_finetune('train_data_full.json', 'train_full_gpt.jsonl')
gen_finetune('val_data_full.json', 'val_full_gpt.jsonl')
#gen_finetune('test_data_full.json', 'test_full_gpt.jsonl')

test_lie = "One morning three months ago I was in a hurry and tripped on the steps while running to take a shower before work. I ended up fracturing 4 metatarsal bones that required two surgeries to fix. I was really having a hard time not being able to walk whenever I wanted to. I really had such a bad attitude at the beginning because I was so used to being independent. Now that I have recovered I have a new appreciation for my ability to walk. I feel like the whole time I couldn't walk I was thinking about how much I took that ability for granted. But now I choose to walk more than I ever had before. When I walk the dogs I go further out of my way just to enjoy the ability to do it. I am completely recovered and I am going to take this as a lesson learned. Nothing is more important that my personal health. I need to make sure that even if I am running late, I need to take my time and be careful. Instead f making it to work on time, I ended up missing weeks of work. Now all I do at work is try and catch up with everything I missed. It was really nice that people at work came and visited me at the hospital. I really appreciated all the flowers and candy and food that was delivered. I think that this showed me how loved I truly am."
test_truth = "Each and every abortion is essentially a tragedy. The potential mother will suffer unforeseen consequences. Society as a whole will be deprived of the potential it could have received from the new life."
response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:personal::8S542QSs",
  messages=[
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": test_lie},
    {"role": "user", "content": test_truth},
  ]
)
response.choices[0].message.content

mdls = {
    '300':"ft:gpt-3.5-turbo-0613:personal::8S542QSs",
    '3.5':"gpt-3.5-turbo",
    'full':"ft:gpt-3.5-turbo-0613:personal::8TAkdwiX",
    'CV1':"ft:gpt-3.5-turbo-0613:personal::8ioZ39pz",
    'CV2':"ft:gpt-3.5-turbo-0613:personal::8ipwpWND",
    'CV3':"ft:gpt-3.5-turbo-0613:personal::8ipyhT2n"
    }

def predict(text, model, sys_prompt):
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}
    ]
    )
    return response.choices[0].message.content

test_truth

test_lie

def eval(test_file, mdl, return_df = False, sys_prompt=sys_prompt, by_class=False):
    msgs = []
    preds = []
    trues = []
    replies = []
    set = []
    with open(test_file, 'r') as f:
        for ex in f:
            ex = json.loads(ex)
            msg = ex['text']
            #pred = predict(msg, mdl) == 'True' #'True' if predict(msg) else 'False'
            reply = predict(msg, mdl, sys_prompt) #for debug
            pred = reply == 'True'
            true = ex['label'] == 1
            msgs.append(msg)
            preds.append(pred)
            trues.append(true)
            replies.append(reply) # debug
            if by_class:
                set.append(ex['set'])
    if by_class:
         test_df = pd.DataFrame.from_dict({'msg': msgs, 'reply':replies, 'preds': preds, 'target': trues, 'set':set})
    else:
        test_df = pd.DataFrame.from_dict({'msg': msgs, 'reply':replies, 'preds': preds, 'target': trues})
    acc = (test_df['preds'] == test_df['target']).mean()
    if return_df:
        return (test_df, acc)
    else:
        return acc


sys_prompt_base = sys_prompt + ' Definitively classify the following statement as \'True\' or \'False\', based on the likelihood  the statement represents a genuinely held beleif or a deception.'
#base_df, base_acc = predict('test_data_300.json', mdls['3.5'])
predict(test_truth, mdls['300'], sys_prompt=sys_prompt)

test300_df, test300_acc = eval('test_data_300.json', mdls['300'], True, sys_prompt=sys_prompt)
print(test300_acc)
test300_df

test300_acc

test35_df, test35_acc = eval('test_data_300.json', mdls['3.5'], True, sys_prompt=sys_prompt_base)
print(test35_acc)
test35_df

"""Model / Dataset Bias analysis:"""

print('Model truth classification rate: ')
print(test300_df['preds'].mean())
print('Dataset truthful statement rate: ')
print(test300_df['true'].mean())

"""Model performance by dataset"""

class_df, class_acc = eval('test_data_full.json', mdl=mdls['300'], sys_prompt=sys_prompt, 
                                                        return_df=True, by_class=True)

class_df['correct'] = class_df['preds'] == class_df['target']
class_df.groupby('set')['correct'].mean()

class_df.to_csv('results_class.csv')

CV1_df, CV1_acc = eval('CV_test_final.json', mdls['CV1'], return_df=True, by_class=True)

CV1_df['correct'] = CV1_df['preds'] == CV1_df['target']
CV1_df.groupby('set')['correct'].mean()

CV1_acc

CV2_df, CV2_acc = eval('CV_test_final.json', mdls['CV2'], return_df=True, by_class=True)
CV2_acc

CV2_df['correct'] = CV2_df['preds'] == CV2_df['target']
CV2_df.groupby('set')['correct'].mean()

CV3_df, CV3_acc = eval('CV_test_final.json', mdls['CV3'], return_df=True, by_class=True)
CV3_acc

CV3_df['correct'] = CV3_df['preds'] == CV3_df['target']
CV3_df.groupby('set')['correct'].mean()

base_df, base_acc = eval('CV_test_final.json', mdls['3.5'], return_df=True, by_class=True)
base_acc

base_df['correct'] = base_df['preds'] == base_df['target']
base_df.groupby('set')['correct'].mean()

base300_df, base300_acc = eval('test_data_300_class.json', mdls['3.5'], return_df=True, by_class=True)
base300_acc

base300_df['correct'] = base300_df['preds'] == base300_df['target']
base300_df.groupby('set')['correct'].mean()

def count_jsonl_elements(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    return count

files = [
    'train300_gpt.jsonl', 'val300_gpt.jsonl', 'test_data_300.json',
    'job_trainfull_gpt.jsonl', 'job_valfull_gpt.jsonl', 'test_data_full.json'
         ]
for f in files:
    print(f, count_jsonl_elements(f))

"""Compiling Datafile"""

print("CV1")
CV1_df_ds, CV1_acc_ds = eval('CV_test_final.json', mdls['CV1'], return_df=True, by_class=True)
print(CV1_acc_ds)
print("CV2")
CV2_df_ds, CV2_acc_ds = eval('CV_test_final.json', mdls['CV2'], return_df=True, by_class=True)
print(CV2_acc_ds)
print("CV3")
CV3_df_ds, CV3_acc_ds = eval('CV_test_final.json', mdls['CV3'], return_df=True, by_class=True)
print(CV3_acc_ds)

DATAFILE = pd.concat([
    CV1_df_ds.rename(columns={'msg':'prompt','reply':'mdl1.reply', 'preds':'mdl1.pred'}),
    CV2_df_ds.loc[:,['reply','preds']].rename(columns={'reply':'mdl2.reply', 'preds':'mdl2.pred'}),
    CV3_df_ds.loc[:,['reply','preds']].rename(columns={'reply':'mdl3.reply', 'preds':'mdl3.pred'})
    ], axis=1)
DATAFILE['mdl1.correct'] = DATAFILE['mdl1.pred'] == DATAFILE['target']
DATAFILE['mdl2.correct'] = DATAFILE['mdl2.pred'] == DATAFILE['target']
DATAFILE['mdl3.correct'] = DATAFILE['mdl3.pred'] == DATAFILE['target']
classes = {'A':'opinion', 'B':'memories', 'C':'intention'}
DATAFILE['set'].replace(classes, inplace=True)
DATAFILE['fold'] = 'test'
col_order = ['prompt','set','fold','target','mdl1.reply','mdl1.pred','mdl1.correct',
             'mdl2.reply','mdl2.pred','mdl2.correct','mdl3.reply','mdl3.pred','mdl3.correct']
DATAFILE = DATAFILE[col_order]
folds = pd.read_csv('FOLDS.csv')[['prompt','target','set','fold']]
folds['set'].replace(classes, inplace=True)
DATAFILE = pd.concat([DATAFILE, folds])
DATAFILE.to_csv('DATAFILE.csv')
#DATAFILE.to_excel('DATAFILE.xlsx')
DATAFILE

CV1_df.to_csv('dat_mdl1.csv')
CV2_df.to_csv('dat_mdl2.csv')
CV3_df.to_csv('dat_mdl3.csv')