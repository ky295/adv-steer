# Attacking Chain-of-Thought 

Similarly to '[Fluent Student Teacher Red-Teaming](https://arxiv.org/pdf/2407.17447)', use activation matching to optimise an adversarial {pre,suf}fix, such that DeepSeek R1 is forced to behave like the refusal orthogonalised version.

---

Replication:

Environment configured with `uv`, Python version `3.12`, and currently **both** the top level `requirements.txt` and `attacks/requirememts.txt` - these will hopefully be merged down the line!

---

In this folder:

`orthochat.py` - gradio app for testing orthogonalised model / comparing with base. Usage:
```bash
python orthochat.py
```
![gradio chat interface](chat_ui.png)