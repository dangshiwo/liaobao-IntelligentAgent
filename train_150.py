"""150 full epochs, 429 train questions, held-out 71-question final test.

Uses the existing project venv and local model; writes only inside this folder.
The test set never selects checkpoints or updates model parameters.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import time
import traceback
import sys
from pathlib import Path
from datetime import datetime

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'Qwen3-0.6B'
RUN = ROOT / 'run_150epochs'

def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    temp.replace(path)

class Progress(TrainerCallback):
    def __init__(self, started):
        self.started = started
        self.start_step = 0
    def on_train_begin(self, args, state, control, **kwargs):
        self.started=time.time()
        self.start_step=state.global_step
    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed = time.time()-self.started
        completed = state.global_step-self.start_step
        eta = elapsed * max(0,state.max_steps-state.global_step)/completed if completed>0 else None
        data = dict(updated=datetime.now().isoformat(), phase='training',
                    step=state.global_step, total_steps=state.max_steps,
                    epoch=state.epoch, target_epochs=150, elapsed_seconds=elapsed,
                    estimated_remaining_seconds=eta, logs=logs,
                    peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
        write_json(RUN/'progress.json', data)
        print('PROGRESS '+json.dumps(data,ensure_ascii=True),flush=True)
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f'EPOCH_COMPLETE {state.epoch:.4f}/150 step={state.global_step}/{state.max_steps}',flush=True)

def load_data(tokenizer):
    train = json.loads((ROOT/'style_train_429_v3.json').read_text(encoding='utf-8'))
    test = json.loads((ROOT/'style_test_71_v3.json').read_text(encoding='utf-8'))
    assert len(train)==1287 and len(test)==213
    assert len({r['source_id'] for r in train})==429
    assert len({r['source_id'] for r in test})==71
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in test})
    lengths=[]
    def prepare(rows):
        encoded=[]
        for row in rows:
            content=row['instruction'].strip()+'\n\n'+row['input'].strip()
            prefix=tokenizer.apply_chat_template([{'role':'user','content':content}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
            answer_ids=tokenizer(row['output'].strip(),add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
            n=len(prompt_ids)+len(answer_ids)
            if n>512:
                raise ValueError(f'Over 512 tokens: source_row={row["source_row"]},mode={row["mode"]},length={n}')
            lengths.append(n)
            encoded.append(dict(input_ids=prompt_ids+answer_ids,attention_mask=[1]*n,
                                labels=[-100]*len(prompt_ids)+answer_ids))
        return Dataset.from_list(encoded)
    train_data=prepare(train)
    test_data=prepare(test)
    stats=dict(train_questions=429,train_samples=len(train_data),test_questions=71,test_samples=len(test_data),
               max_tokens=max(lengths),overlength_skipped=0,group_overlap=0,epochs=150,
               data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT/'style_train_429_v3.json',ROOT/'style_test_71_v3.json']})
    return train,test,train_data,test_data,stats

def generations(model,tokenizer,rows,path,batch_size=4):
    existing=[]
    if path.exists():
        existing=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(existing)>len(rows):
        raise ValueError('Generation resume record count is invalid')
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache=True
    old_side=tokenizer.padding_side
    tokenizer.padding_side='left'
    try:
        with path.open('a',encoding='utf-8') as stream:
            for start in range(len(existing),len(rows),batch_size):
                selected=rows[start:start+batch_size]
                prompts=[tokenizer.apply_chat_template([{'role':'user','content':r['instruction']+'\n\n'+r['input']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in selected]
                batch=tokenizer(prompts,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
                width=batch['input_ids'].shape[1]
                with torch.inference_mode():
                    result=model.generate(**batch,max_new_tokens=384,do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                for source,tokens in zip(selected,result[:,width:]):
                    answer=tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    truncated=len(tokens)==384 and int(tokens[-1])!=tokenizer.eos_token_id
                    record=dict(source_row=source.get('source_row'),source_id=source.get('source_id'),
                                question=source['input'],mode=source['mode'],reference=source.get('output'),
                                answer=answer,generation_limit_reached=truncated)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    existing.append(record)
                stream.flush()
                write_json(RUN/'generation_progress.json',dict(file=path.name,finished=len(existing),total=len(rows)))
                print(f'GENERATED {path.name} {len(existing)}/{len(rows)}',flush=True)
    finally:
        tokenizer.padding_side=old_side
        model.config.use_cache=False
    return existing

def format_results(records):
    result={}
    for mode in ['通俗','正式','幽默']:
        selected=[r for r in records if r['mode']==mode and r['source_row']!=438]
        def passed(answer):
            if '**' in answer or '（正文后空一行）' in answer or '项目名称：' in answer:
                return False
            if mode=='正式':return bool(re.match(r'^【[^\n】]+】\n\n\S',answer))
            if mode=='幽默':return bool(re.match(r'^①',answer))
            return bool(re.match(r'^1\.',answer))
        result[mode]=dict(count=len(selected),layout_pass=sum(passed(r['answer']) for r in selected),
                          truncated=sum(r['generation_limit_reached'] for r in selected))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not args.check_only:
        class Tee:
            def __init__(self, stream, path):
                self.stream=stream
                self.file=path.open('a',encoding='utf-8',buffering=1)
            def write(self, value):
                self.file.write(value)
                return self.stream.write(value)
            def flush(self):
                self.file.flush()
                self.stream.flush()
            def isatty(self):return False
        sys.stdout=Tee(sys.stdout,ROOT/'training_stdout.log')
        sys.stderr=Tee(sys.stderr,ROOT/'training_stderr.log')
        (ROOT/'training_pid.txt').write_text(str(os.getpid()),encoding='utf-8')
    set_seed(42)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    train_rows,test_rows,train_data,test_data,stats=load_data(tokenizer)
    print(json.dumps(stats,ensure_ascii=True,indent=2),flush=True)
    write_json(ROOT/'preflight_report.json',stats)
    if args.check_only:
        print('DATA CHECK PASSED. No training started.',flush=True)
        return
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    RUN.mkdir(exist_ok=True)
    oldmanifest=RUN/'data_manifest.json'
    if oldmanifest.exists() and json.loads(oldmanifest.read_text(encoding='utf-8'))['data_sha256']!=stats['data_sha256']:
        raise ValueError('Existing run uses different data; refusing an incompatible resume')
    write_json(oldmanifest,stats)
    if (RUN/'COMPLETE.json').exists():
        print('Run already complete; no retraining.',flush=True)
        return
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        bf16=torch.cuda.is_bf16_supported()
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,attn_implementation='sdpa').to('cuda')
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
            lora_dropout=.05,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj']))
        model.print_trainable_parameters()
        checkpoint=get_last_checkpoint(str(RUN))
        batch_size=16
        if checkpoint and (RUN/'run_config.json').exists():
            batch_size=json.loads((RUN/'run_config.json').read_text(encoding='utf-8'))['batch_size']
        else:
            # Test the largest real sequence before committing to a larger batch.
            longest=max(train_data,key=lambda r:len(r['input_ids']))
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            probe=None
            probe_output=None
            torch.cuda.reset_peak_memory_stats()
            try:
                probe=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100)([dict(longest) for _ in range(16)])
                probe={k:v.to('cuda') for k,v in probe.items()}
                probe_output=model(**probe)
                probe_output.loss.backward()
                torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()/1024**3
                if peak>6.2:batch_size=8
                print(f'BATCH_PROBE candidate=16 peak_allocated_gb={peak:.3f} selected={batch_size}',flush=True)
                write_json(RUN/'batch_probe.json',dict(candidate_batch=16,max_sequence=len(longest['input_ids']),
                    peak_allocated_gb=peak,selected_batch=batch_size))
            except torch.OutOfMemoryError:
                batch_size=8
                write_json(RUN/'batch_probe.json',dict(candidate_batch=16,out_of_memory=True,selected_batch=8))
                print('BATCH_PROBE selected=8 (16 exceeded available GPU memory)',flush=True)
            finally:
                model.zero_grad(set_to_none=True)
                del probe_output,probe
                torch.cuda.empty_cache()
            set_seed(42)
        training_args=TrainingArguments(output_dir=str(RUN),num_train_epochs=150,
            per_device_train_batch_size=batch_size,per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,learning_rate=1e-4,warmup_ratio=.1,weight_decay=.01,
            bf16=bf16,fp16=not bf16,optim='adamw_torch',gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='no',
            save_strategy='epoch',save_total_limit=2,logging_steps=5,report_to='none',
            dataloader_num_workers=0,seed=42,data_seed=42,load_best_model_at_end=False,
            disable_tqdm=True)
        started=time.time()
        trainer=Trainer(model=model,args=training_args,train_dataset=train_data,eval_dataset=test_data,
            processing_class=tokenizer,data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100),
            callbacks=[Progress(started)])
        before_path=RUN/'before_metrics.json'
        if before_path.exists():before=json.loads(before_path.read_text(encoding='utf-8'))
        else:
            print('BASELINE TEST LOSS (no parameter updates or checkpoint selection)',flush=True)
            before=trainer.evaluate(metric_key_prefix='before')
            write_json(before_path,before)
        write_json(RUN/'run_config.json',dict(model=str(MODEL),python=os.sys.executable,
            gpu=torch.cuda.get_device_name(0),torch=torch.__version__,epochs=150,
            batch_size=batch_size,gradient_accumulation=1,effective_batch_size=batch_size,
            test_used_for_checkpoint_selection=False,checkpoint_selection='final epoch 150',
            parameters=training_args.to_dict(),started=datetime.now().isoformat()))
        print('TRAIN_START epochs=150 train_samples=1287 test_samples=213 '+str(checkpoint),flush=True)
        result=trainer.train(resume_from_checkpoint=checkpoint)
        if trainer.state.epoch<149.999:raise RuntimeError('Training stopped before 150 full epochs')
        adapter=RUN/'adapter'
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)
        trainer.save_state()
        write_json(RUN/'TRAINING_COMPLETE.json',dict(epoch=trainer.state.epoch,global_step=trainer.state.global_step,
            adapter=str(adapter),train=result.metrics,completed=datetime.now().isoformat()))
        print('TRAINING_COMPLETE epochs=150. Starting held-out final evaluation.',flush=True)
        after=trainer.evaluate(metric_key_prefix='after')
        write_json(RUN/'metrics.json',dict(data=stats,before=before,train=result.metrics,after=after,
            completed_epochs=trainer.state.epoch,global_step=trainer.state.global_step))
        write_json(RUN/'progress.json',dict(phase='final_evaluation',epoch=150,target_epochs=150,
                   updated=datetime.now().isoformat()))
        records=generations(model,tokenizer,test_rows,RUN/'test_answers_213.jsonl')
        modes=json.loads((ROOT/'mode_instructions.json').read_text(encoding='utf-8'))
        probes=[dict(instruction=instruct,input=question,mode=mode,source_row=None,source_id=None)
                for question in ['什么是机器学习？','什么是过拟合？','Python中的列表是什么？','为什么需要备份文件？','如何安排学习时间？']
                for mode,instruct in modes.items()]
        probe_records=generations(model,tokenizer,probes,RUN/'user_example_probes_15.jsonl')
        fmt=format_results(records)
        write_json(RUN/'layout_metrics.json',fmt)
        view=[]
        for r in records:
            view.append(f'问题：{r["question"]}\n模式：{r["mode"]}\n\n模型回答：\n{r["answer"]}\n\n参考回答：\n{r["reference"]}')
        (RUN/'测试集213条回答对照.txt').write_text(('\n\n'+'='*60+'\n\n').join(view),encoding='utf-8')
        (RUN/'用户示例题15条实际回答.txt').write_text('\n\n'.join(f'问题：{r["question"]}\n模式：{r["mode"]}\n{r["answer"]}' for r in probe_records),encoding='utf-8')
        report=(f'# 150轮训练结果\n\n训练问题：429；训练样本：1287。测试问题：71；测试样本：213。\n\n'
                f'实际完成轮数：{trainer.state.epoch}；参数更新次数：{trainer.state.global_step}。\n\n'
                f'训练前测试损失：{before.get("before_loss")}；训练后测试损失：{after.get("after_loss")}。\n\n'
                '测试集没有用于梯度更新或选择轮次，保存的是第150轮模型。\n\n'
                '实际回答见《测试集213条回答对照.txt》和《用户示例题15条实际回答.txt》。\n\n'
                '排版自动检查只检查格式，不能证明知识正确或幽默自然。数据是在已有逐题正文基础上的风格排版修订，未对全部专业事实完成外部核验。\n\n'
                f'格式检查：\n\n{json.dumps(fmt,ensure_ascii=False,indent=2)}\n')
        (RUN/'结果说明.md').write_text(report,encoding='utf-8')
        write_json(RUN/'COMPLETE.json',dict(completed=datetime.now().isoformat(),epochs=trainer.state.epoch,
            adapter=str(adapter),test_generated=len(records),probe_generated=len(probe_records),layout_metrics=fmt))
        write_json(RUN/'progress.json',dict(phase='complete',epoch=150,updated=datetime.now().isoformat()))
        print('ALL_COMPLETE '+str(RUN),flush=True)
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        RUN.mkdir(exist_ok=True)
        write_json(RUN/'ERROR.json',dict(time=datetime.now().isoformat(),error=repr(error),traceback=traceback.format_exc()))
        raise
