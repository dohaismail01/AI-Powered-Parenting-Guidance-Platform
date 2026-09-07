from llama_cpp import Llama

llm = Llama(
    model_path=r"C:\Users\PCM\.cache\huggingface\hub\models--dohaiismail--nile-chat-parenting-lora-gguf\snapshots\8f9b48a8928065777599cc367f943e2762ad6f84\Nile-Chat-4B.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=True
)

print("تم التحميل بنجاح!")