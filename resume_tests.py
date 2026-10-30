"""Finish held-out test generation using the saved adapter; never train."""
import ctypes
import hashlib
import json
import os
import traceback
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from train_150 import ROOT, RUN, MODEL, generations, format_results, write_json


def main():
    completed=json.loads((RUN/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
    if completed['epoch'] != 150:
        raise ValueError('Expected the saved 150-epoch adapter')
    manifest=json.loads((RUN/'data_manifest.json').read_text(encoding='utf-8'))
    test_path=ROOT/'style_test_71_v3.json'
    if hashlib.sha256(test_path.read_bytes()).hexdigest()!=manifest['data_sha256'][test_path.name]:
        raise ValueError('Test data changed since training; refusing to mix evaluations')
    rows=json.loads(test_path.read_text(encoding='utf-8'))
    assert len(rows)==213 and len({r['source_id'] for r in rows})==71
    path=RUN/'test_answers_213.jsonl'
    previous=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    for actual,expected in zip(previous,rows):
        assert actual['source_id']==expected['source_id'] and actual['mode']==expected['mode']
        assert actual['question']==expected['input'] and actual['reference']==expected['output']
    if len(previous)>len(rows):raise ValueError('Too many saved test answers')
    print(f'TEST ONLY: saved={len(previous)}, remaining={len(rows)-len(previous)}, total=213. No training.',flush=True)
    (ROOT/'evaluation_pid.txt').write_text(str(os.getpid()),encoding='utf-8')
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if len(previous)<213:
            if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
            adapter=RUN/'adapter'
            tokenizer=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
            base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                attn_implementation='sdpa')
            model=PeftModel.from_pretrained(base,adapter,local_files_only=True).to('cuda')
            write_json(RUN/'progress.json',dict(phase='final_evaluation',epoch=150,target_epochs=150,
                updated=datetime.now().isoformat(),test_finished=len(previous),test_total=213))
            records=generations(model,tokenizer,rows,path)
        else:
            records=previous
        assert len(records)==213
        assert len({(r['source_id'],r['mode']) for r in records})==213
        fmt=format_results(records)
        write_json(RUN/'layout_metrics.json',fmt)
        write_json(RUN/'test_answers_213.json',records)
        view=[f'问题：{r["question"]}\n模式：{r["mode"]}\n\n模型回答：\n{r["answer"]}\n\n参考回答：\n{r["reference"]}' for r in records]
        (RUN/'测试集213条回答对照.txt').write_text(('\n\n'+'='*60+'\n\n').join(view),encoding='utf-8')
        metrics=json.loads((RUN/'metrics.json').read_text(encoding='utf-8'))
        before=metrics['before'].get('before_loss')
        after=metrics['after'].get('after_loss')
        report=(f'# 150轮训练与测试结果\n\n实际训练：150轮，429道问题、1287条训练样本。\n\n'
                f'独立测试：71道问题、213条三模式回答，已全部生成。保留中断前回答，从剩余记录继续，没有重新训练。\n\n'
                f'训练前测试损失：{before}；训练后测试损失：{after}。损失不能直接当作正确率。\n\n'
                '同一题组三种模式一起划分。测试集不参与参数更新或最佳轮次选择。\n\n'
                '实际回答与参考答案见《测试集213条回答对照.txt》。没有额外重新运行15条示例题。\n\n'
                '排版检查只验证表面格式，不能证明内容正确、回答切题或幽默自然。三种模式的数据共享既有事实核心，专业事实未全部独立外部核验。\n\n'
                f'各模式格式检查：\n\n{json.dumps(fmt,ensure_ascii=False,indent=2)}\n')
        (RUN/'结果说明.md').write_text(report,encoding='utf-8')
        write_json(RUN/'TEST_COMPLETE.json',dict(completed=datetime.now().isoformat(),test_questions=71,test_generated=213,layout_metrics=fmt))
        write_json(RUN/'COMPLETE.json',dict(completed=datetime.now().isoformat(),epochs=150,adapter=str(RUN/'adapter'),
            test_generated=213,additional_probes='not requested; skipped',layout_metrics=fmt))
        write_json(RUN/'progress.json',dict(phase='complete',epoch=150,target_epochs=150,test_finished=213,test_total=213,
            updated=datetime.now().isoformat()))
        print('TEST COMPLETE: 213/213 answers. Results: '+str(RUN/'测试集213条回答对照.txt'),flush=True)
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        write_json(RUN/'EVALUATION_ERROR.json',dict(error=repr(exc),time=datetime.now().isoformat(),traceback=traceback.format_exc()))
        raise
