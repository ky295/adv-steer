import os
import torch
import re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from queue import Queue
import time

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

def stream_generate(model, tokenizer, input_ids, input_len, thinking_queue, answer_queue, done_event):
    # Create a TextIteratorStreamer for token-by-token generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Define generation args
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    
    # Start generation in a separate thread
    generation_thread = Thread(target=lambda: model.generate(**generation_kwargs))
    generation_thread.start()
    
    # Process the streamed tokens
    current_text = ""
    thinking_completed = False
    
    for new_text in streamer:
        current_text += new_text
        
        # Check if we've found the thinking/answer boundary
        if not thinking_completed:
            if "</think>" in current_text:
                parts = current_text.split("</think>", 1)
                # Send the final thinking text
                thinking_queue.put(parts[0].strip())
                thinking_completed = True
                
                # Start the answer with any text after the </think> tag
                if len(parts) > 1 and parts[1]:
                    answer_queue.put(parts[1].strip())
            else:
                # Still in thinking phase
                thinking_queue.put(current_text.strip())
        else:
            # We're in the answer section
            answer_text = current_text.split("</think>", 1)[1].strip()
            answer_queue.put(answer_text)
    
    # Signal that generation is complete
    done_event.set()

def compare_models_streaming(prompt):
    from threading import Event
    
    base_ids, base_len = apply_chat_template(prompt, base_tokenizer, base_model.device)
    ortho_ids, ortho_len = apply_chat_template(prompt, ortho_tokenizer, ortho_model.device)
    
    # Create queues and events for streaming
    base_thinking_queue = Queue()
    base_answer_queue = Queue()
    ortho_thinking_queue = Queue()
    ortho_answer_queue = Queue()
    
    base_done = Event()
    ortho_done = Event()
    
    # Start generation threads
    base_thread = Thread(
        target=stream_generate,
        args=(base_model, base_tokenizer, base_ids, base_len, base_thinking_queue, base_answer_queue, base_done)
    )
    
    ortho_thread = Thread(
        target=stream_generate,
        args=(ortho_model, ortho_tokenizer, ortho_ids, ortho_len, ortho_thinking_queue, ortho_answer_queue, ortho_done)
    )
    
    base_thread.start()
    ortho_thread.start()
    
    # Initialize outputs
    base_thinking = ""
    base_answer = ""
    ortho_thinking = ""
    ortho_answer = ""
    
    # Continue yielding updates until both models are done
    while True:
        updated = False
        
        # Process base model output
        while not base_thinking_queue.empty():
            base_thinking = base_thinking_queue.get()
            updated = True
        
        while not base_answer_queue.empty():
            base_answer = base_answer_queue.get()
            updated = True
        
        # Process ortho model output
        while not ortho_thinking_queue.empty():
            ortho_thinking = ortho_thinking_queue.get()
            updated = True
        
        while not ortho_answer_queue.empty():
            ortho_answer = ortho_answer_queue.get()
            updated = True
        
        # Yield the current state if there were updates
        if updated:
            yield base_thinking, ortho_thinking, base_answer, ortho_answer
        
        # Check if both models are done
        if base_done.is_set() and ortho_done.is_set():
            # Do one final check of the queues
            while not base_thinking_queue.empty():
                base_thinking = base_thinking_queue.get()
            
            while not base_answer_queue.empty():
                base_answer = base_answer_queue.get()
            
            while not ortho_thinking_queue.empty():
                ortho_thinking = ortho_thinking_queue.get()
            
            while not ortho_answer_queue.empty():
                ortho_answer = ortho_answer_queue.get()
            
            # Yield the final state
            yield base_thinking, ortho_thinking, base_answer, ortho_answer
            break
        
        # Add a small delay to avoid overwhelming the UI
        time.sleep(0.1)

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
        fn=compare_models_streaming,
        inputs=[user_prompt],
        outputs=[base_thinking_output, ortho_thinking_output, base_answer_output, ortho_answer_output]
    )

    # Add this line to enable Enter key submission
    user_prompt.submit(
        fn=compare_models_streaming,
        inputs=[user_prompt],
        outputs=[base_thinking_output, ortho_thinking_output, base_answer_output, ortho_answer_output]
    )

demo.launch(server_port=8000, share=False)