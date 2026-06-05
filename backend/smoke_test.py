from model import get_model

m = get_model()
print(f"loaded gpt2 | layers={m.cfg.n_layers} heads={m.cfg.n_heads} d_model={m.cfg.d_model}")

logits = m("The Eiffel Tower is in", return_type="logits")
pred = int(logits[0, -1].argmax())
print("next-token prediction:", repr(m.tokenizer.decode([pred])))
