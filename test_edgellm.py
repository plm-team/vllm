from vllm import LLM, SamplingParams

llm = LLM(model="/home/edgellm/elm-eval/EdgeLLM-v2-ckpts-decay-6/hf_iter_162122", trust_remote_code=True, max_model_len=4096, enforce_eager=True)
# llm = LLM(model="/home/edgellm/models/MiniCPM3-4B", trust_remote_code=True, max_model_len=4096)
# llm = LLM(model="/home/edgellm/models/Nemotron-Mini-4B-Instruct")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "What is machine learning"
]
sampling_params = SamplingParams(temperature=0.7, top_p=0.8)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
