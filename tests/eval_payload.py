import json
from typing import List, Optional
from pydantic import BaseModel

# Define the models exactly as in your FastAPI app.
class RetrievalEvaluationItem(BaseModel):
    query: str
    ground_truth_docs: List[str]
    retrieved_docs: List[str]
    ground_truth_answer: Optional[str] = None
    system_answer: Optional[str] = None

class EvaluationRequest(BaseModel):
    data: List[RetrievalEvaluationItem]

# You can either load your evaluation data from a source or simulate it.
def generate_sample_evaluation_data() -> EvaluationRequest:
    sample_data = [
        {
            "query": "Jak se připojit k eduroam na ZČU?",
            "ground_truth_docs": ["Eduroam.md", "Eduroam_Kde_se_připojit3F.md"],
            "retrieved_docs": [
                "Eduroam.md",
                "Eduroam_Kde_se_připojit3F.md",
                "English_explanation_of_UWB_network_and_IT_usage_rules.md"
            ],
            "ground_truth_answer": (
                "Pro připojení k eduroam použijte automatický konfigurační nástroj, který je dostupný "
                "pro Windows, macOS, Linux, Android a iOS. Přihlašovací údaje (login a heslo) jsou uvedeny "
                "v dokumentaci eduroam a na stránkách ZČU."
            ),
            "system_answer": (
                "K připojení k eduroam využijte automatický konfigurační nástroj a zadejte své přihlašovací "
                "údaje podle pokynů na stránkách univerzity."
            )
        },
        {
            "query": "Jak kontaktovat HelpDesk CIV?",
            "ground_truth_docs": ["HelpDesk.md"],
            "retrieved_docs": ["HelpDesk.md", "První_krůčky_2011.pdf"],
            "ground_truth_answer": (
                "HelpDesk CIV můžete kontaktovat e-mailem na adrese helpdesk@zcu.cz, telefonicky na čísle "
                "+420 377 63 8888 nebo osobně na adrese Univerzitní 20, 301 00 Plzeň."
            ),
            "system_answer": (
                "Kontaktujte HelpDesk CIV e-mailem, telefonicky nebo osobně na Univerzitě v Plzni "
                "(Univerzitní 20, 301 00 Plzeň)."
            )
        },
        {
            "query": "Jak se přihlásit do cloudových služeb ZČU?",
            "ground_truth_docs": ["CloudovC3A9_sluČeby.md"],
            "retrieved_docs": ["CloudovC3A9_sluČeby.md"],
            "ground_truth_answer": (
                "Přihlášení do cloudových služeb ZČU se provádí přes jednotné přihlášení pomocí webového rozhraní "
                "Sunstone, kde uživatel zadá své přihlašovací údaje přes SSO."
            ),
            "system_answer": (
                "Pro přihlášení do cloudových služeb ZČU využijte webové rozhraní Sunstone a přihlašte se pomocí "
                "jednotného přihlášení (SSO)."
            )
        },
        {
            "query": "Jak vytvořit nový kalendář v Google kalendáři?",
            "ground_truth_docs": ["Google_Calendar.md"],
            "retrieved_docs": ["Google_Calendar.md"],
            "ground_truth_answer": (
                "Pro vytvoření nového kalendáře klikněte na tlačítko 'Vytvořit nový kalendář', vyplňte název, popis, "
                "umístění, časové pásmo a nastavení sdílení, a poté klikněte na 'Vytvořit kalendář'. Nový kalendář se objeví "
                "v sekci 'Moje kalendáře'."
            ),
            "system_answer": (
                "Klikněte na 'Vytvořit nový kalendář', vyplňte potřebné údaje a potvrďte vytvoření kalendáře; poté se vám "
                "zobrazí v sekci 'Moje kalendáře'."
            )
        },
        {
            "query": "Jaké jsou minimální požadavky na počítač pro nákup ICT na ZČU?",
            "ground_truth_docs": ["NC3A1kup_ICT.md"],
            "retrieved_docs": ["NC3A1kup_ICT.md"],
            "ground_truth_answer": (
                "Minimální požadavky zahrnují procesor s Passmark skóre nad 21000, alespoň 6 jader, 16 GB DDR5 RAM, "
                "integrovanou grafiku, SSD disk o kapacitě minimálně 512 GB, a dostatečný počet USB portů (minimálně 6, "
                "z toho alespoň 2 USB 3.0)."
            ),
            "system_answer": (
                "Počítač musí mít procesor s minimálně 21000 body v Passmark, 6 jader, 16 GB DDR5, SSD disk 512GB a další "
                "specifikované porty a funkce, jak je uvedeno v dokumentu o nákupu ICT."
            )
        },
        {
            "query": "Jak použít funkci PrintScreen?",
            "ground_truth_docs": ["NC3A1vod_na_použití_funkce_printscreen.md"],
            "retrieved_docs": ["NC3A1vod_na_použití_funkce_printscreen.md"],
            "ground_truth_answer": (
                "Po zobrazení chyby stiskněte tlačítko Print Scrn, otevřete program Malování, vložte obrázek pomocí "
                "klávesové zkratky Ctrl+V a poté soubor uložte a odešlete jako přílohu na helpdesk@zcu.cz."
            ),
            "system_answer": (
                "Stiskněte tlačítko Print Scrn, otevřete program Malování, vložte obrázek (Ctrl+V), uložte soubor a "
                "odešlete ho na helpdesk@zcu.cz."
            )
        },
        {
            "query": "Jak získat Orion konto na ZČU?",
            "ground_truth_docs": ["První_krůčky_2011.pdf"],
            "retrieved_docs": ["První_krůčky_2011.pdf"],
            "ground_truth_answer": (
                "Získání Orion konta zahrnuje vyzvednutí JIS karty, registraci pomocí webové aplikace na registrace.zcu.cz, "
                "ověření funkčnosti konta a aktivaci konta během 24 až 36 hodin."
            ),
            "system_answer": (
                "Nejprve si vyzvedněte JIS kartu, poté se zaregistrujte na registrace.zcu.cz a ověřte funkčnost svého Orion "
                "konta, které bude aktivní do 36 hodin."
            )
        }
    ]
    return EvaluationRequest(data=sample_data)

if __name__ == "__main__":
    # Generate our evaluation payload
    evaluation_payload = generate_sample_evaluation_data()

    # Validate and then serialize to JSON using model_dump()
    payload_json = json.dumps(evaluation_payload.model_dump(), indent=2, ensure_ascii=False)

    # Optionally, save to a file:
    with open("evaluation_payload.json", "w", encoding="utf-8") as f:
        f.write(payload_json)

    print("Generated JSON payload:")
    print(payload_json)
