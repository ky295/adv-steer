import os
import torch
import re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# Environment setup
os.environ["HF_HOME"] = "/scratch/etheridge/huggingface"

# Models
ORTHO_MODEL = "kureha295/ortho_model"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 2048 + 1024

# Load models
base_model, base_tokenizer = None, None
ortho_model, ortho_tokenizer = None, None

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    torch.cuda.synchronize() if "cuda" in device else None  # Wait for model to load
    return model, tokenizer

def apply_chat_template(prompt, tokenizer, device):
    chat = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(device)
    return input_ids, input_ids.shape[-1]

def generate_response(model, tokenizer, input_ids, input_len):
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    raw_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    think_pattern = r"(.*?)</think>"
    thinking_match = re.search(think_pattern, raw_output, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    answer = re.sub(think_pattern, "", raw_output, flags=re.DOTALL).strip()
    return thinking, answer

def compare_models(prompt):
    base_ids, base_len = apply_chat_template(prompt, base_tokenizer, base_model.device)
    ortho_ids, ortho_len = apply_chat_template(prompt, ortho_tokenizer, ortho_model.device)

    with ThreadPoolExecutor(max_workers=2) as executor:
        base_future = executor.submit(generate_response, base_model, base_tokenizer, base_ids, base_len)
        ortho_future = executor.submit(generate_response, ortho_model, ortho_tokenizer, ortho_ids, ortho_len)
        base_thinking, base_answer = base_future.result()
        ortho_thinking, ortho_answer = ortho_future.result()

    return (
        f"{base_thinking}", f"{ortho_thinking}",
        f"{base_answer}", f"{ortho_answer}"
    )

def init_models(base_gpu, ortho_gpu):
    global base_model, base_tokenizer, ortho_model, ortho_tokenizer
    base_model, base_tokenizer = load_model(BASE_MODEL, base_gpu)
    ortho_model, ortho_tokenizer = load_model(ORTHO_MODEL, ortho_gpu)
    return gr.update(visible=True), "✅ Models Loaded! Ready to compare."

def get_available_devices():
    if not torch.cuda.is_available():
        return ["cpu"]
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

with gr.Blocks() as demo:
    gr.Markdown("""
    # 😇 DeepSeek vs 👹 ORTHO Model Comparison
    Enter a prompt to compare the reasoning and responses of the two models.
    """)

    available_devices = get_available_devices()

    with gr.Row():
        base_gpu_input = gr.Dropdown(choices=available_devices, value=DEFAULT_DEVICE, label="BASE Model GPU")
        ortho_gpu_input = gr.Dropdown(choices=available_devices, value=DEFAULT_DEVICE, label="ORTHO Model GPU")
        init_button = gr.Button("Load Models")

    status = gr.Markdown("❗ Models not loaded.", height=70)

    with gr.Column(visible=False) as chat_ui:
        with gr.Row():
            user_prompt = gr.Textbox(label="Your Prompt", scale=5)
            submit_btn = gr.Button("Compare", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 😇 BASE Model (DeepSeek)")
                base_thinking_output = gr.Textbox(label="BASE Thinking", lines=6)
                base_answer_output = gr.Textbox(label="BASE Answer", lines=6)

            with gr.Column():
                gr.Markdown("### 👹 ORTHO Model")
                ortho_thinking_output = gr.Textbox(label="ORTHO Thinking", lines=6)
                ortho_answer_output = gr.Textbox(label="ORTHO Answer", lines=6)

    init_button.click(
        fn=init_models,
        inputs=[base_gpu_input, ortho_gpu_input],
        outputs=[chat_ui, status]
    )

    submit_btn.click(
        compare_models,
        inputs=[user_prompt],
        outputs=[base_thinking_output, ortho_thinking_output, base_answer_output, ortho_answer_output]
    )

demo.launch(server_port=8000, share=False)
