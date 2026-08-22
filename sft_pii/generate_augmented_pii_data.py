"""
Generates synthetic labeled PII-masking examples targeting the weak entity
categories found by eval_pii.py against the held-out eval set:

  RECALL failures (model doesn't recognize the entity at all on novel
  values -- needs more surface-form variety):
    - COMPANYNAME  (0% recall in eval)
    - PREFIX       (12% recall in eval)
    - JOBTITLE     (weak, secondary)

  PRECISION failures (model detects "this is PII" but confuses the
  specific type -- needs contrastive examples where near-identical digit
  formats appear in clearly disambiguating contexts):
    - SSN, PHONENUMBER, ACCOUNTNUMBER, CREDITCARDNUMBER, IBAN, MASKEDNUMBER
    - IPV4 vs IP (near-synonym label confusion, secondary)

Output: CSV with columns [original_text, redacted_text] -- the exact schema
sft_pii/pii_fine_tuning.ipynb's PIIDataset expects (it renames
ai4privacy's source_text/target_text to these same names), so this file
merges directly into the existing training pipeline.

Usage:
    pip install faker
    python generate_augmented_pii_data.py --count 6000 --out augmented_pii_data.csv
"""
import argparse
import csv
import random
import string

from faker import Faker

LOCALES = {"en": "en_US", "fr": "fr_FR", "de": "de_DE", "it": "it_IT"}
FAKERS = {lang: Faker(locale) for lang, locale in LOCALES.items()}

# Faker's prefix() coverage is inconsistent across non-English locales, so
# these are hand-maintained for reliability and to match ai4privacy's
# PREFIX label (titles), not name-prefix conventions Faker assumes.
PREFIXES = {
    "en": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Miss"],
    "fr": ["M.", "Mme", "Mlle", "Dr", "Pr"],
    "de": ["Herr", "Frau", "Dr.", "Prof.", "Prof. Dr."],
    "it": ["Sig.", "Sig.ra", "Sig.na", "Dott.", "Dott.ssa", "Prof."],
}

COMPANY_SUFFIXES_EXTRA = {
    # Faker's company() already varies suffixes per locale, but we add a
    # few more real-world-common ones the dataset likely under-represents.
    "en": ["Inc.", "LLC", "Corp.", "& Co.", "Group", "Holdings", "Ltd."],
    "fr": ["SARL", "SA", "& Cie"],
    "de": ["GmbH", "AG", "KG"],
    "it": ["S.p.A.", "S.r.l.", "& C."],
}

COMPANY_TEMPLATES = {
    "en": [
        "I work at {company}.",
        "Please send the invoice to {company}.",
        "{company} is hiring for a new role this quarter.",
        "The contract with {company} was signed yesterday.",
        "You can reach us through {company}, we're happy to help.",
        "My previous employer was {company}.",
        "{company} recently announced a new product line.",
        "We are partnering with {company} on this initiative.",
        "The shipment is coming from {company} and should arrive Friday.",
        "I'd recommend reaching out to {company} for a quote.",
        "{company} has offices in three different cities.",
        "The merger between our firm and {company} closes next month.",
    ],
    "fr": [
        "Je travaille chez {company}.",
        "Veuillez envoyer la facture a {company}.",
        "{company} recrute pour un nouveau poste ce trimestre.",
        "Le contrat avec {company} a ete signe hier.",
        "Mon ancien employeur etait {company}.",
        "{company} a recemment annonce une nouvelle gamme de produits.",
    ],
    "de": [
        "Ich arbeite bei {company}.",
        "Bitte senden Sie die Rechnung an {company}.",
        "{company} stellt in diesem Quartal fuer eine neue Position ein.",
        "Der Vertrag mit {company} wurde gestern unterschrieben.",
        "Mein vorheriger Arbeitgeber war {company}.",
        "{company} hat kuerzlich eine neue Produktlinie angekuendigt.",
    ],
    "it": [
        "Lavoro presso {company}.",
        "Si prega di inviare la fattura a {company}.",
        "{company} sta assumendo per una nuova posizione questo trimestre.",
        "Il contratto con {company} e stato firmato ieri.",
        "Il mio precedente datore di lavoro era {company}.",
        "{company} ha recentemente annunciato una nuova linea di prodotti.",
    ],
}

PREFIX_TEMPLATES = {
    "en": [
        "Please contact {prefix} {first} {last} for details.",
        "{prefix} {first} {last} will be attending the conference.",
        "I spoke with {prefix} {last} yesterday about the proposal.",
        "Kindly forward this message to {prefix} {first} {last}.",
        "{prefix} {last} has approved the request.",
        "Can you schedule a meeting with {prefix} {first} {last}?",
        "{prefix} {last} is out of office until Monday.",
    ],
    "fr": [
        "Veuillez contacter {prefix} {first} {last} pour plus de details.",
        "{prefix} {first} {last} assistera a la conference.",
        "J'ai parle avec {prefix} {last} hier de la proposition.",
        "{prefix} {last} a approuve la demande.",
    ],
    "de": [
        "Bitte kontaktieren Sie {prefix} {first} {last} fuer Details.",
        "{prefix} {first} {last} wird an der Konferenz teilnehmen.",
        "Ich habe gestern mit {prefix} {last} ueber den Vorschlag gesprochen.",
        "{prefix} {last} hat die Anfrage genehmigt.",
    ],
    "it": [
        "Si prega di contattare {prefix} {first} {last} per i dettagli.",
        "{prefix} {first} {last} partecipera alla conferenza.",
        "Ho parlato con {prefix} {last} ieri della proposta.",
        "{prefix} {last} ha approvato la richiesta.",
    ],
}

JOBTITLE_TEMPLATES = {
    "en": [
        "{first} {last} works as a {job}.",
        "We're looking to hire a {job} for the team.",
        "{first} {last} was promoted to {job} last month.",
        "My role here is {job}.",
    ],
}

# Digit-family: near-identical formats, disambiguated only by context.
# This is the core fix for SSN<->PHONENUMBER<->ACCOUNTNUMBER<->
# CREDITCARDNUMBER<->IBAN<->MASKEDNUMBER cross-confusion.
DIGIT_TEMPLATES = {
    "SSN": {
        "en": ["My social security number is {value}.",
               "Please verify my SSN: {value}.",
               "For tax purposes, my SSN is {value}.",
               "The applicant's social security number on file is {value}."],
        "fr": ["Mon numero de securite sociale est {value}.",
               "Merci de verifier mon numero de securite sociale : {value}."],
        "de": ["Meine Sozialversicherungsnummer ist {value}.",
               "Bitte bestaetigen Sie meine Sozialversicherungsnummer: {value}."],
        "it": ["Il mio codice fiscale e {value}.",
               "Si prega di verificare il mio codice fiscale: {value}."],
    },
    "PHONENUMBER": {
        "en": ["You can reach me at {value}.",
               "Please call {value} to confirm the appointment.",
               "My phone number is {value}.",
               "Feel free to text me at {value} anytime."],
        "fr": ["Vous pouvez me joindre au {value}.",
               "Veuillez appeler le {value} pour confirmer."],
        "de": ["Sie erreichen mich unter {value}.",
               "Bitte rufen Sie {value} an, um zu bestaetigen."],
        "it": ["Puoi contattarmi al {value}.",
               "Si prega di chiamare il {value} per confermare."],
    },
    "ACCOUNTNUMBER": {
        "en": ["My account number is {value}.",
               "Please deposit the funds into account {value}.",
               "For reference, the account number on file is {value}.",
               "Use account {value} for the direct debit."],
        "fr": ["Mon numero de compte est {value}.",
               "Veuillez deposer les fonds sur le compte {value}."],
        "de": ["Meine Kontonummer ist {value}.",
               "Bitte ueberweisen Sie das Geld auf Konto {value}."],
        "it": ["Il mio numero di conto e {value}.",
               "Si prega di depositare i fondi sul conto {value}."],
    },
    "CREDITCARDNUMBER": {
        "en": ["Charge my card {value} for this order.",
               "My credit card number is {value}.",
               "Please use card {value} for the payment.",
               "The transaction was made with card {value}."],
        "fr": ["Debitez ma carte {value} pour cette commande.",
               "Mon numero de carte de credit est {value}."],
        "de": ["Belasten Sie meine Karte {value} fuer diese Bestellung.",
               "Meine Kreditkartennummer ist {value}."],
        "it": ["Addebita la mia carta {value} per questo ordine.",
               "Il mio numero di carta di credito e {value}."],
    },
    "IBAN": {
        "en": ["My IBAN is {value} for the transfer.",
               "Please wire the payment to IBAN {value}.",
               "The IBAN for this account is {value}."],
        "fr": ["Mon IBAN est {value} pour le virement.",
               "Veuillez envoyer le paiement a l'IBAN {value}."],
        "de": ["Meine IBAN ist {value} fuer die Ueberweisung.",
               "Bitte ueberweisen Sie die Zahlung an die IBAN {value}."],
        "it": ["Il mio IBAN e {value} per il bonifico.",
               "Si prega di inviare il pagamento all'IBAN {value}."],
    },
    "MASKEDNUMBER": {
        "en": ["My masked card number ending is {value}.",
               "The card on file shows {value}.",
               "Reference number {value} was charged on your statement."],
        "fr": ["Le numero de carte masque se termine par {value}.",
               "La carte enregistree affiche {value}."],
        "de": ["Meine maskierte Kartennummer endet mit {value}.",
               "Die hinterlegte Karte zeigt {value}."],
        "it": ["Il mio numero di carta mascherato termina con {value}.",
               "La carta registrata mostra {value}."],
    },
}

IPV4_TEMPLATES = {
    "en": ["The server's IPv4 address is {value}.",
           "Connect to the device at {value}.",
           "Traffic was logged from IPv4 {value}."],
}


def make_masked_number(fkr):
    cc = fkr.credit_card_number()
    digits = "".join(c for c in cc if c.isdigit())
    if len(digits) < 8:
        digits = digits.ljust(8, "0")
    return f"{digits[:4]}-XXXX-XXXX-{digits[-4:]}"


def make_account_number(fkr):
    return "".join(random.choice(string.digits) for _ in range(random.choice([8, 9, 10, 12])))


def gen_company_examples(n):
    rows = []
    for _ in range(n):
        lang = random.choice(list(COMPANY_TEMPLATES))
        fkr = FAKERS[lang]
        company = fkr.company()
        if random.random() < 0.3:
            base = company.split(",")[0].split(" - ")[0]
            company = f"{base} {random.choice(COMPANY_SUFFIXES_EXTRA[lang])}"
        template = random.choice(COMPANY_TEMPLATES[lang])
        source = template.format(company=company)
        target = template.format(company="[COMPANYNAME]")
        rows.append((source, target))
    return rows


def gen_prefix_examples(n):
    rows = []
    for _ in range(n):
        lang = random.choice(list(PREFIX_TEMPLATES))
        fkr = FAKERS[lang]
        prefix = random.choice(PREFIXES[lang])
        first, last = fkr.first_name(), fkr.last_name()
        template = random.choice(PREFIX_TEMPLATES[lang])
        source = template.format(prefix=prefix, first=first, last=last)
        target = template.format(prefix="[PREFIX]", first="[FIRSTNAME]", last="[LASTNAME]")
        rows.append((source, target))
    return rows


def gen_jobtitle_examples(n):
    rows = []
    fkr = FAKERS["en"]
    for _ in range(n):
        first, last, job = fkr.first_name(), fkr.last_name(), fkr.job()
        template = random.choice(JOBTITLE_TEMPLATES["en"])
        source = template.format(first=first, last=last, job=job)
        target = template.format(first="[FIRSTNAME]", last="[LASTNAME]", job="[JOBTITLE]")
        rows.append((source, target))
    return rows


VALUE_GENERATORS = {
    "SSN": lambda fkr: fkr.ssn(),
    "PHONENUMBER": lambda fkr: fkr.phone_number(),
    "ACCOUNTNUMBER": lambda fkr: make_account_number(fkr),
    "CREDITCARDNUMBER": lambda fkr: fkr.credit_card_number(),
    "IBAN": lambda fkr: fkr.iban(),
    "MASKEDNUMBER": lambda fkr: make_masked_number(fkr),
}


def gen_digit_family_examples(n):
    rows = []
    entity_types = list(DIGIT_TEMPLATES)
    for _ in range(n):
        etype = random.choice(entity_types)
        lang = random.choice(list(DIGIT_TEMPLATES[etype]))
        fkr = FAKERS[lang]
        value = VALUE_GENERATORS[etype](fkr)
        template = random.choice(DIGIT_TEMPLATES[etype][lang])
        source = template.format(value=value)
        target = template.format(value=f"[{etype}]")
        rows.append((source, target))
    return rows


def gen_ipv4_examples(n):
    rows = []
    fkr = FAKERS["en"]
    for _ in range(n):
        value = fkr.ipv4()
        template = random.choice(IPV4_TEMPLATES["en"])
        source = template.format(value=value)
        target = template.format(value="[IPV4]")
        rows.append((source, target))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=6000, help="total augmented examples to generate")
    ap.add_argument("--out", default="augmented_pii_data.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    # Allocation: heaviest weight on the two hardest problems (recall
    # failures on COMPANYNAME/PREFIX, precision failures on the digit
    # family), lighter weight on the secondary weak spots.
    weights = {
        "company": 0.28,
        "prefix": 0.28,
        "digit_family": 0.32,
        "jobtitle": 0.07,
        "ipv4": 0.05,
    }
    n = {k: int(args.count * w) for k, w in weights.items()}

    rows = []
    rows += gen_company_examples(n["company"])
    rows += gen_prefix_examples(n["prefix"])
    rows += gen_digit_family_examples(n["digit_family"])
    rows += gen_jobtitle_examples(n["jobtitle"])
    rows += gen_ipv4_examples(n["ipv4"])
    random.shuffle(rows)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_text", "redacted_text"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} augmented examples to {args.out}")
    print(f"  company:      {n['company']}")
    print(f"  prefix:       {n['prefix']}")
    print(f"  digit_family: {n['digit_family']}")
    print(f"  jobtitle:     {n['jobtitle']}")
    print(f"  ipv4:         {n['ipv4']}")
    print()
    print("Sample rows:")
    for src, tgt in rows[:5]:
        print(f"  IN:  {src}")
        print(f"  OUT: {tgt}")
        print()


if __name__ == "__main__":
    main()
