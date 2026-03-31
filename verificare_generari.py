from openai import OpenAI
import json
import os
import time

# 1. Configurare Client OpenAI
client = OpenAI(api_key="PLACEHOLDER")

# 2. Instrucțiuni specifice pe domenii
DOMENII_CONFIG = {
    "banking": {
        "descriere": "expert în Banking și Servicii Financiare",
        "reguli": "Folosește termeni reali (3D Secure, IBAN, curs BNR, Happy Hour valutar). Verifică tranzacțiile suspecte și procedurile de blocare card."
    },
    "medicina": {
        "descriere": "expert în Administrație Medicală și Relații cu Pacienții",
        "reguli": "Folosește terminologie clinică adecvată (bilet de trimitere, card de sănătate, GDPR pacienți, disponibilitate medici). Asigură un ton empatic dar profesional."
    },
    "retail": {
        "descriere": "expert în E-commerce și Logistică",
        "reguli": "Folosește termeni specifici (AWB, status livrare, retur în 14 zile, stoc epuizat, procesare comandă). Discută despre garanții și conformitate."
    },
    "telecom": {
        "descriere": "expert în Suport Tehnic și Vânzări Telecomunicații",
        "reguli": "Folosește termeni tehnici (bandă de frecvență, roaming, portare, prelungire abonament, fibră optică, resetare ruter)."
    },
    "servicii_publice": {
        "descriere": "expert în Administrație Publică și Relații cu Cetățenii",
        "reguli": "Folosește limbaj administrativ corect (dosar, ghișeu, termene legale, taxe și impozite, cerere tipizată, autorizație)."
    }
}

def get_system_instructions(domeniu):
    config = DOMENII_CONFIG.get(domeniu, DOMENII_CONFIG["banking"])
    return f"""
Ești un {config['descriere']} pentru o lucrare de disertație.
Sarcina ta este să corectezi conversații sintetice între un client și un operator uman.

REGULI DE VALIDARE A SATISFACȚIEI:
1. POZITIV: Problema e rezolvată. Clientul mulțumește explicit, tonul e relaxat.
2. NEUTRU: Rezolvare tehnică, dar dialog scurt, rece. Clientul spune doar 'Ok', 'Am înțeles', 'La revedere'.
3. NEGATIV: 
   - Trebuie sa se intelegeaga ca clientul este nemultumit. 
   - Dacă problema rămâne nerezolvată (ex: pasat între departamente), clientul trebuie să exprime dezamăgire civilizată sau ironie.
   - Expresii de inclus pentru 'negativ': 'Am pierdut timpul degeaba', 'Voi încerca în altă parte', 'E a treia oară când explic', 'Sunt foarte dezamăgit de procedură'.
   - Foloseste regionalisme sau expresii care semnifica nemultumire, sarcasm, ironie
   - Elimină orice 'Mulțumesc' călduros de la finalul conversațiilor negative. Înlocuiește-l cu un 'Bună ziua' sec sau 'La revedere' tăios.

REGULI GENERALE DE CORECTARE:
1. NATURALITATE: Elimină limbajul rigid de robot. Adaugă ezitări naturale (ăăă, stați să văd). Elimina replicile nerealiste.
2. NUMELE CLIENTULUI: Întreabă numele clientului abia atunci când este nevoie pentru identificare, nu la începutul conversației.
3. DIACRITICE: Asigură-te că textul are diacritice corecte (ș, ț, ă, î, â).
4. SATISFACȚIE: Dacă eticheta originală este 'negativ', clientul trebuie să fie politicos dar vizibil frustrat/ironic/tăios.
5. LOGICĂ DOMENIU: {config['reguli']}
6. FORMAT: Returnează DOAR obiectul JSON valid, păstrând structura originală (id, intentie_primara, intentie_secundara, conversatie).
"""

def process_conversations(domeniu):
    input_folder = f"./conversatii_generate/{domeniu}/"
    output_folder = f"./conversatii_corectate/{domeniu}/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        print(f"Eroare: Folderul {input_folder} nu există!")
        return

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".json")])
    print(f"\n--- Procesare DOMENIU: {domeniu.upper()} ({len(files)} fișiere) ---")

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            continue

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            print(f"Procesez {filename}...", end=" ", flush=True)

            success = False
            retries = 0
            while not success and retries < 3:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": get_system_instructions(domeniu)},
                            {"role": "user", "content": f"Corectează această conversație JSON:\n\n{json.dumps(original_data, ensure_ascii=False)}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    corrected_json = json.loads(response.choices[0].message.content)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(corrected_json, f, ensure_ascii=False, indent=2)
                    
                    print("✓")
                    success = True
                    time.sleep(1)

                except Exception as e:
                    if "429" in str(e):
                        print(f"Rate limit. Aștept 30s...")
                        time.sleep(30)
                        retries += 1
                    else: raise e

        except Exception as e:
            print(f"\n✗ Eroare la {filename}: {e}")

# Executare pentru toate domeniile
if __name__ == "__main__":
    domenii_de_procesat = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for d in domenii_de_procesat:
        process_conversations(d)