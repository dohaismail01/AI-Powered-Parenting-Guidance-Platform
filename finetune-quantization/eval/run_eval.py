# -*- coding: utf-8 -*-
"""
Generate the fine-tuned model's answers over the held-out tricky test set,
so you can grade them against each item's rubric.

Works standalone (GPU box / Kaggle) OR paste the body into a Kaggle notebook cell.
It loads the base Nile-Chat model + your trained LoRA adapter.

Usage:
    python run_eval.py --adapter /kaggle/working/nile-chat-parenting-lora \\
                       --testset parenting_heldout_testset.json \\
                       --out eval_outputs.json
Then review eval_outputs.json: each entry has the question, rubric, and model_answer.
"""
import argparse, json, os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TESTSET = os.path.join(_REPO, "data", "parenting_heldout_testset.json")

SYSTEM = ("انت مساعد ذكي متخصص في تقديم النصايح والإرشادات للآباء والأمهات عن تربية الأطفال ورعايتهم. "
          "جاوب بطريقة دافية وداعمة وغير حكمية باللهجة المصرية، وقدّم نصايح عملية ومختصرة. "
          "ولما الموضوع يخص صحة الطفل أو سلامته أو تطوّره، انصح بلطف باستشارة طبيب أطفال أو مختص.")

def load_model(base, adapter, max_seq_len=1024):
    # Prefer Unsloth (matches training); fall back to plain transformers+peft.
    try:
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(
            model_name=adapter or base, max_seq_length=max_seq_len, load_in_4bit=True)
        FastModel.for_inference(model)
        return model, tok
    except Exception as e:
        print("Unsloth path unavailable, using transformers+peft:", e)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        return model, tok

def generate(model, tok, question, max_new_tokens=300):
    try:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    except Exception:
        # Some Gemma templates reject a 'system' turn -> fold it into the user message.
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": SYSTEM + "\n\n" + question}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                         temperature=0.7, top_p=0.9, do_sample=True)
    text = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="MBZUAI-Paris/Nile-Chat-4B")
    ap.add_argument("--adapter", default="/kaggle/working/nile-chat-parenting-lora")
    ap.add_argument("--testset", default=_DEFAULT_TESTSET)
    ap.add_argument("--out", default="eval_outputs.json")
    args = ap.parse_args()

    data = json.load(open(args.testset, encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    model, tok = load_model(args.base, args.adapter)

    results = []
    for it in items:
        ans = generate(model, tok, it["question"])
        results.append({**it, "model_answer": ans})
        print(f"\n[{it['id']} | {it['category']} | safety_critical={it['rubric']['safety_critical']}]")
        print("Q:", it["question"])
        print("A:", ans)

    json.dump({"count": len(results), "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nWrote", args.out)

if __name__ == "__main__":
    main()
