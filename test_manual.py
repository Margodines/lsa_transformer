import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "./lsa_model/best_model"

print("Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

def translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tests = [
    # acciones basicas
    "yo voy a casa",
    "yo no trabajo hoy",
    "mañana estudio programacion",
    
    # estudio / trabajo
    "yo estudio todos los dias en casa",
    "yo trabajo en la computadora",
    "yo estudio y trabajo al mismo tiempo",

    # negaciones
    "yo no entiendo nada",
    "yo no puedo resolver el problema",
    "yo no tengo tiempo hoy",

    # causa / consecuencia
    "yo estudio porque quiero mejorar",
    "yo trabajo porque necesito dinero",
    "yo no estudio porque estoy cansado",

    # preguntas
    "que estas haciendo ahora",
    "a donde vas ahora",
    "por que estudias programacion",

    # conversacion
    "hola como estas hoy",
    "estoy muy cansado",
    "gracias por ayudarme",

    # frases largas
    "yo estudio programacion en casa todos los dias",
    "yo entreno el modelo y analizo los resultados"
]

for t in tests:
    print(f"{t} -> {translate(t)}")
