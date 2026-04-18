"""
Test pentru Gemini pe task de rezumat.
"""
import os
import time

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("Seteaza GEMINI_API_KEY!")
    exit(1)

from google import genai
from google.genai import types
client = genai.Client(api_key=GEMINI_API_KEY)

dialog_test = (
    "CLIENT: Buna ziua, am o problema cu cardul meu blocat.\n"
    "OPERATOR: Buna ziua, va ajut imediat. Puteti sa imi dati numarul cardului?\n"
    "CLIENT: Da, este 1234 5678 9012 3456.\n"
    "OPERATOR: Am verificat, cardul a fost blocat din cauza unor tranzactii suspecte. "
    "Va pot debloca acum, dar va trebui sa schimbati PIN-ul.\n"
    "CLIENT: Bine, va rog sa il deblocati.\n"
    "OPERATOR: Gata, cardul este activ. Veti primi un SMS pentru schimbarea PIN-ului.\n"
    "CLIENT: Multumesc frumos, chiar m-ati ajutat!"
)

teste = [
    {
        "nume": "Prompt simplu rezumat — max_tokens=150",
        "prompt": f"Rezuma urmatoarea conversatie telefonica in limba romana.\n\nConversatie:\n{dialog_test}\n\nRezumat:",
        "max_tokens": 150
    },
    {
        "nume": "Prompt simplu rezumat — max_tokens=500",
        "prompt": f"Rezuma urmatoarea conversatie telefonica in limba romana.\n\nConversatie:\n{dialog_test}\n\nRezumat:",
        "max_tokens": 500
    },
    {
        "nume": "Prompt structurat V2 — max_tokens=150",
        "prompt": (
            "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n"
            "Genereaza un rezumat de tip SCURT al conversatiei de mai jos.\n\n"
            "CERINTE:\n"
            "- 1-2 propozitii, 20-40 cuvinte\n"
            "- Scrie in limba romana\n"
            "- Mentioneaza problema principala si rezultatul final\n\n"
            f"Conversatie:\n{dialog_test}\n\n"
            "Rezumat SCURT:"
        ),
        "max_tokens": 150
    },
    {
        "nume": "Fara max_tokens (default)",
        "prompt": f"Rezuma urmatoarea conversatie telefonica in limba romana in 2-3 propozitii.\n\nConversatie:\n{dialog_test}\n\nRezumat:",
        "max_tokens": None
    },
]

print("=== TEST GEMINI REZUMAT ===\n")
for test in teste:
    print(f"Test: {test['nume']}")
    try:
        config_kwargs = {}
        if test["max_tokens"]:
            config_kwargs["max_output_tokens"] = test["max_tokens"]

        start = time.time()
        raspuns = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=test["prompt"],
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        ):
            if chunk.text:
                raspuns += chunk.text
        latenta = time.time() - start

        print(f"  Raspuns: '{raspuns}'")
        print(f"  Lungime: {len(raspuns.split())} cuvinte")
        print(f"  Latenta: {latenta:.3f}s")
    except Exception as e:
        print(f"  EROARE: {e}")
    print()