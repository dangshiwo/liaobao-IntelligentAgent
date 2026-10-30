"""Interactive Q&A with the final 150-epoch adapter. Does not start training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run_150epochs'
MODEL = ROOT / 'models' / 'Qwen3-0.6B'

def main():
    adapter = RUN / 'adapter'
    if not (RUN / 'TRAINING_COMPLETE.json').exists():
        print('150轮训练尚未完成，请稍后再运行。')
        if (RUN / 'progress.json').exists():
            state = json.loads((RUN / 'progress.json').read_text(encoding='utf-8'))
            print(f'状态：{state.get("phase")}；轮数：{state.get("epoch")} / 150')
        if (RUN / 'ERROR.json').exists():
            print(f'发现异常记录，请检查：{RUN / "ERROR.json"}')
        return
    if not (adapter / 'adapter_config.json').exists():
        raise FileNotFoundError(f'找不到本次训练的adapter：{adapter}')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('当前Python环境无法使用CUDA GPU')
    modes = json.loads((ROOT / 'mode_instructions.json').read_text(encoding='utf-8'))
    names = {'1':'通俗','2':'正式','3':'幽默'}
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, adapter, local_files_only=True).to('cuda')
    model.eval()
    model.config.use_cache = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    print(f'已加载150轮训练结果：{adapter}')
    print('每个问题独立回答。输入 exit 退出。')
    while True:
        try:
            question = input('\n问题 > ').strip()
            if question.lower() in {'exit','quit'}:break
            if not question:continue
            choice = input('选择模式：1通俗 / 2正式 / 3幽默 > ').strip()
            if choice not in names:
                print('请输入1、2或3。请重新输入问题并选择模式。')
                continue
        except (EOFError,KeyboardInterrupt):break
        mode = names[choice]
        prompt = tokenizer.apply_chat_template([
            {'role':'user','content':modes[mode]+'\n\n'+question}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs = tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
        if inputs['input_ids'].shape[1]>1024:
            print('问题过长，请缩短后重试。')
            continue
        with torch.inference_mode():
            result = model.generate(**inputs,max_new_tokens=512,do_sample=False,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        tokens = result[0,inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(tokens,skip_special_tokens=True).strip()
        print('\n'+answer)
        if len(tokens)==512 and int(tokens[-1])!=tokenizer.eos_token_id:
            print('[达到生成长度上限，回答可能未完成]')
        with (ROOT/'interactive_answers.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(question=question,mode=mode,answer=answer,
                                        adapter=str(adapter)),ensure_ascii=False)+'\n')

if __name__=='__main__':
    main()
