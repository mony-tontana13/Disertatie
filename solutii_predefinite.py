"""
Solutii predefinite per intentie pentru robotul telefonic.

SOLUTII_PREDEFINITE:
  - text  -> raspuns generic, se citeste direct prin TTS
  - None  -> raspuns personalizat, generat de LLM pe baza conversatiei
  - "OPERATOR" -> redirectare catre operator uman

DETALII_NECESARE:
  Lista de detalii pe care robotul trebuie sa le colecteze
  inainte de a oferi solutia.
"""

SOLUTII_PREDEFINITE = {

    # ─── BANKING ──────────────────────────────────────────────────────────────
    "problema_credit":              None,           # LLM — depinde de rata specifica
    "tranzactie_gresita":           None,           # LLM — depinde de suma si data
    "card_blocat": (
        "Am verificat statusul cardului dumneavoastra. Cardul a fost blocat automat "
        "ca masura de securitate. Pentru deblocare, va rugam sa vizitati cea mai "
        "apropiata sucursala cu actul de identitate, sau sa apelati linia de urgenta "
        "carduri la numarul de pe spatele cardului. Deblocarea dureaza aproximativ "
        "10 minute."
    ),
    "tranzactie_suspecta": (
        "Am blocat temporar contul pentru protectia dumneavoastra. Echipa de "
        "securitate va analiza tranzactiile suspecte in urmatoarele 2 ore. Veti fi "
        "contactat telefonic pentru confirmare. Daca nu recunoasteti tranzactia, "
        "aceasta va fi anulata si suma returnata in maxim 3 zile lucratoare."
    ),
    "problema_transfer":            None,           # LLM — depinde de suma si destinatar
    "problema_schimb_valutar":      None,           # LLM — depinde de cursul aplicat
    "problema_sold": (
        "Am verificat istoricul contului dumneavoastra. Soldul reflecta toate "
        "tranzactiile procesate pana in acest moment. Tranzactiile in curs de "
        "procesare pot aparea cu intarziere de pana la 24 de ore. Va rugam sa "
        "verificati si aplicatia bancii pentru detalii in timp real. Daca "
        "neconcordanta persista, un consilier va analiza situatia in detaliu."
    ),
    "card_pierdut": (
        "Am blocat imediat cardul pentru a preveni orice utilizare neautorizata. "
        "Un card nou va fi emis si livrat la adresa din contract in 3 pana la 5 "
        "zile lucratoare. Alternativ, il puteti ridica de la sucursala in aceeasi "
        "zi. Pana la primirea noului card, puteti efectua plati prin aplicatia "
        "bancii sau portofelul mobil. Nu exista taxe pentru reemitere."
    ),

    # ─── MEDICINA ─────────────────────────────────────────────────────────────
    "rezultate_analize":            None,           # LLM — depinde de tipul analizelor
    "problema_reteta":              None,           # LLM — depinde de medicament si data
    "problema_asigurare": (
        "Am verificat dosarul dumneavoastra de asigurare. Serviciile solicitate "
        "sunt acoperite conform politei dumneavoastra. Documentele necesare pentru "
        "decontare trebuie trimise la adresa asiguratorului in termen de 60 de zile "
        "de la data serviciului medical. Formularul de decontare este disponibil "
        "pe site-ul clinicii si la receptie."
    ),
    "reclamatie_personal":          None,           # LLM — depinde de incident specific
    "consultatie_anulata":          None,           # LLM — depinde de data/ora reprogramare
    "problema_facturare":           None,           # LLM — depinde de suma contestata
    "problema_programare":          None,           # LLM — depinde de data/ora
    "anulare_programare": (
        "Programarea dumneavoastra a fost anulata cu succes. Veti primi o confirmare "
        "prin SMS in cateva minute. Daca doriti sa reprogramati, va stam la "
        "dispozitie acum sau puteti suna oricand in timpul programului. Nu exista "
        "penalizari pentru anularea cu cel putin 24 de ore inainte."
    ),

    # ─── RETAIL ───────────────────────────────────────────────────────────────
    "produs_lipsa_stoc":            None,           # LLM — depinde de produs specific
    "comanda_gresita": (
        "Ne cerem scuze pentru eroarea de expediere. Am initiat procesul de retur "
        "pentru produsul gresit. Un curier va prelua coletul de la adresa "
        "dumneavoastra in urmatoarele 24 de ore, fara costuri din partea "
        "dumneavoastra. Produsul corect va fi expediat imediat dupa confirmarea "
        "returului, cu livrare prioritara."
    ),
    "problema_livrare": (
        "Am verificat statusul comenzii dumneavoastra. Am identificat o intarziere "
        "la curier si am deschis un incident prioritar. Comanda va fi livrata in "
        "urmatoarele 24 de ore. Daca livrarea nu se realizeaza in acest termen, "
        "veti primi automat un voucher de compensatie de 20 de lei pentru "
        "urmatoarea comanda."
    ),
    "problema_garantie": (
        "Produsul dumneavoastra se afla in perioada de garantie. Am inregistrat "
        "solicitarea de service cu numarul de caz pe care vi-l vom trimite prin "
        "SMS. Un tehnician va contacta in 24 de ore pentru a stabili modalitatea "
        "de remediere. Daca produsul nu poate fi reparat, il vom inlocui sau "
        "va vom returna contravaloarea."
    ),
    "reclamatie_produs":            None,           # LLM — depinde de problema specifica
    "anulare_comanda": (
        "Comanda dumneavoastra a fost anulata cu succes. Suma platita va fi "
        "returnata in contul sau pe cardul folosit la plata in termen de 3 pana "
        "la 5 zile lucratoare, in functie de banca. Veti primi o confirmare prin "
        "email cu detaliile rambursarii."
    ),
    "comanda_intarziata": (
        "Comanda dumneavoastra a inregistrat o intarziere din cauza unui volum "
        "mare de comenzi. Noul termen estimat de livrare este de 1 pana la 2 "
        "zile lucratoare. Puteti urmari statusul in timp real prin link-ul de "
        "tracking trimis pe email. Ne cerem scuze pentru neinconvenient si va "
        "oferim un discount de 10 la suta la urmatoarea comanda."
    ),
    "retur_produs": (
        "Am initiat procesul de retur pentru produsul dumneavoastra. Un curier "
        "va prelua coletul de la adresa de livrare in urmatoarele 24 de ore. "
        "Va rugam sa ambalati produsul in cutia originala si sa includeti bonul "
        "fiscal. Rambursarea va fi procesata in 3 pana la 5 zile lucratoare "
        "dupa primirea si verificarea produsului."
    ),

    # ─── TELECOM ──────────────────────────────────────────────────────────────
    "problema_modificare_abonament": None,          # LLM — depinde de pachetul dorit
    "portare_esuata": (
        "Am verificat solicitarea de portare. Am identificat si corectat eroarea "
        "din sistem. Portarea va fi reactivata in urmatoarele 2 ore. Veti primi "
        "un SMS de confirmare cand numarul va fi activ la noul operator."
    ),
    "problema_internet": (
        "Am verificat conexiunea dumneavoastra in sistem. Am identificat o problema "
        "tehnica in zona care afecteaza mai multi clienti. Echipa tehnica lucreaza "
        "la remediere, iar serviciul va fi restabilit in maximum 2 ore. Veti primi "
        "un SMS la normalizarea conexiunii. Ne cerem scuze pentru inconvenient."
    ),
    "problema_roaming":             None,           # LLM — depinde de tara
    "factura_gresita":              None,           # LLM — depinde de suma contestata
    "reziliere_contract": (
        "Am inregistrat solicitarea de reziliere. Conform contractului, perioada "
        "de preaviz este de 30 de zile. Serviciile vor fi active pana la finalul "
        "acestei perioade. Daca exista o penalitate de reziliere anticipata, "
        "aceasta va fi calculata si comunicata prin email in 24 de ore."
    ),
    "activare_esuata": (
        "Am verificat solicitarea de activare. Am identificat o eroare tehnica "
        "si am escalat cazul catre echipa tehnica. Serviciul va fi activat in "
        "maximum 4 ore. Veti primi un SMS de confirmare. In aceasta perioada "
        "nu veti fi facturat pentru serviciul neactivat."
    ),
    "problema_semnal": (
        "Am verificat acoperirea in zona dumneavoastra. Inregistram o degradare "
        "temporara a semnalului cauzata de lucrari de mentenanta la statia de "
        "baza. Lucrarile vor fi finalizate in urmatoarele 4 ore. Daca problema "
        "persista dupa acest interval, va rugam sa ne contactati din nou."
    ),

    # ─── SERVICII PUBLICE ─────────────────────────────────────────────────────
    "dosar_respins":                None,           # LLM — depinde de motivul respingerii
    "contestatie_decizie": (
        "Am inregistrat contestatia dumneavoastra. Conform legii, termenul de "
        "solutionare este de 30 de zile calendaristice. Veti primi un numar de "
        "inregistrare prin SMS si email. Documentele suport pot fi depuse la "
        "ghiseu sau trimise prin posta recomandata la adresa institutiei."
    ),
    "informatii_program": (
        "Programul de lucru al institutiei este de luni pana vineri, intre orele "
        "8:30 si 16:30. Ghiseele pentru publicul general functioneaza fara "
        "programare intre 8:30 si 12:00. Dupa-amiaza, accesul se face exclusiv "
        "pe baza de programare. Sambata si duminica institutia este inchisa."
    ),
    "reclamatie_serviciu": (
        "Am inregistrat reclamatia dumneavoastra. Institutia are obligatia legala "
        "de a raspunde in 30 de zile calendaristice. Veti primi un numar de "
        "referinta prin SMS. Daca nu primiti raspuns in acest termen, puteti "
        "escalada situatia la institutia ierarhic superioara sau la Avocatul "
        "Poporului."
    ),
    "sesizare_problema": (
        "Sesizarea dumneavoastra a fost inregistrata si va fi transmisa "
        "departamentului competent. Veti primi un numar de inregistrare prin SMS "
        "si puteti urmari statusul pe portalul institutiei. Termenul legal de "
        "raspuns este de 30 de zile."
    ),
    "problema_plata_taxa":          None,           # LLM — depinde de suma si data
    "acte_incomplete":              None,           # LLM — depinde de actele lipsa
    "programare_ghiseu":            None,           # LLM — depinde de data/ora stabilita

    # ─── FALLBACK ─────────────────────────────────────────────────────────────
    "alta_solicitare": "OPERATOR",  # redirect catre operator uman
}

# Detalii necesare per intentie (colectate inainte de solutie)
DETALII_NECESARE = {
    "problema_credit":               ["numarul contractului de credit sau CNP-ul"],
    "tranzactie_gresita":            ["data tranzactiei", "suma tranzactiei"],
    "card_blocat":                   ["ultimele 4 cifre ale cardului"],
    "tranzactie_suspecta":           ["data si suma tranzactiei suspecte"],
    "problema_transfer":             ["data transferului", "suma si IBAN-ul destinatarului"],
    "problema_schimb_valutar":       ["data tranzactiei valutare", "suma convertita"],
    "problema_sold":                 ["data pentru care verificati soldul"],
    "card_pierdut":                  ["ultimele 4 cifre ale cardului pierdut"],
    "rezultate_analize":             ["numele complet", "data recoltarii analizelor"],
    "problema_reteta":               ["numele medicamentului", "data emiterii retetei"],
    "problema_asigurare":            ["numarul politei de asigurare"],
    "reclamatie_personal":           ["numele medicului sau asistentului", "data incidentului"],
    "consultatie_anulata":           ["data si ora consultatiei anulate"],
    "problema_facturare":            ["numarul facturii", "suma contestata"],
    "problema_programare":           ["numele complet", "data programarii existente"],
    "anulare_programare":            ["data si ora programarii de anulat"],
    "produs_lipsa_stoc":             ["numele produsului sau codul de produs"],
    "comanda_gresita":               ["numarul comenzii"],
    "problema_livrare":              ["numarul comenzii"],
    "problema_garantie":             ["numarul comenzii sau seria produsului"],
    "reclamatie_produs":             ["numarul comenzii", "descrierea problemei"],
    "anulare_comanda":               ["numarul comenzii"],
    "comanda_intarziata":            ["numarul comenzii"],
    "retur_produs":                  ["numarul comenzii", "motivul returului"],
    "problema_modificare_abonament": ["numarul de telefon al abonamentului", "pachetul dorit"],
    "portare_esuata":                ["numarul de telefon de portat"],
    "problema_internet":             ["adresa unde se inregistreaza problema"],
    "problema_roaming":              ["tara in care va aflati"],
    "factura_gresita":               ["numarul facturii", "suma contestata"],
    "reziliere_contract":            ["numarul de telefon al contractului"],
    "activare_esuata":               ["numarul de telefon sau codul SIM-ului"],
    "problema_semnal":               ["adresa sau zona cu problema de semnal"],
    "dosar_respins":                 ["numarul dosarului"],
    "contestatie_decizie":           ["numarul deciziei contestate"],
    "informatii_program":            [],
    "reclamatie_serviciu":           ["descrierea problemei", "data incidentului"],
    "sesizare_problema":             ["descrierea problemei"],
    "problema_plata_taxa":           ["data platii", "suma platita"],
    "acte_incomplete":               ["numarul dosarului"],
    "programare_ghiseu":             ["motivul programarii"],
    "alta_solicitare":               [],
}

if __name__ == "__main__":
    generice = [k for k, v in SOLUTII_PREDEFINITE.items() if isinstance(v, str) and v != "OPERATOR"]
    personalizate = [k for k, v in SOLUTII_PREDEFINITE.items() if v is None]
    operator = [k for k, v in SOLUTII_PREDEFINITE.items() if v == "OPERATOR"]
    print(f"GENERIC  (predefinit): {len(generice)}")
    print(f"LLM      (personalizat): {len(personalizate)}")
    print(f"OPERATOR (redirect): {len(operator)}")