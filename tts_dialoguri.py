import asyncio
import edge_tts
import json
import os
import io
from pydub import AudioSegment

# CONFIGURARE
FISIERE_DE_PROCESAT = [
    "conversatii_corectate/banking/BNK_006.json",
    "conversatii_corectate/banking/BNK_008.json",
    "conversatii_corectate/medicina/MED_001.json",
    "conversatii_corectate/medicina/MED_018.json",
    "conversatii_corectate/retail/RET_003.json",
    "conversatii_corectate/retail/RET_010.json",
    "conversatii_corectate/servicii_publice/SP_006.json",
    "conversatii_corectate/servicii_publice/SP_008.json",
    "conversatii_corectate/telecom/TEL_002.json",
    "conversatii_corectate/telecom/TEL_009.json"
]

VOCE_OPERATOR = "ro-RO-AlinaNeural"
VOCE_CLIENT = "ro-RO-EmilNeural"

async def proceseaza_fisier(nume_fisier):
    if not os.path.exists(nume_fisier):
        print(f"Fișierul {nume_fisier} nu a fost găsit. Skip.")
        return

    with open(nume_fisier, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conv_id = data.get("id", "necunoscut")
    replici = data.get("conversatie", [])
    
    print(f"\n Generez conversația completă pentru ID: {conv_id}...")

    # Audio-ul combinat (începem cu un segment gol)
    audio_complet = AudioSegment.empty()
    
    # Adăugăm o mică pauză de liniște între replici (ex: 500ms)
    pauza = AudioSegment.silent(duration=2000)

    for i, replica in enumerate(replici):
        rol = replica["rol"]
        text = replica["text"]
        voce = VOCE_OPERATOR if rol.lower() == "operator" else VOCE_CLIENT
        
        # Generăm audio-ul în memorie (byte stream)
        communicate = edge_tts.Communicate(text, voce)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        # Transformăm byte stream-ul în segment audio pydub
        segment_replica = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        
        # Adăugăm la fișierul final
        audio_complet += segment_replica + pauza
        print(f"   Adăugat replica {i} ({rol})")

    # Salvare fișier final unic
    nume_output = f"conversatie_{conv_id}.mp3"
    audio_complet.export(nume_output, format="mp3")
    print(f"Fișier final salvat: {nume_output}")

async def main():
    if not FISIERE_DE_PROCESAT:
        print("Lista 'FISIERE_DE_PROCESAT' este goală.")
        return

    for fisier in FISIERE_DE_PROCESAT:
        await proceseaza_fisier(fisier)
    
    print("\n Toate conversațiile au fost salvate.")

if __name__ == "__main__":
    asyncio.run(main())