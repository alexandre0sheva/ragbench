"""Generate the bundled hard suite in data/hard: a fictional, license-free corpus and question set built to separate RAG systems.

    python scripts/generate_hard_suite.py          # rewrite data/hard
    python scripts/generate_hard_suite.py --check  # exit 1 if data/hard differs from what this script writes (tests use it)

Everything comes from one seed, so the output is byte-for-byte reproducible. The suite is hard on purpose:

- version_conflict: spec sheets in three revisions and head offices that moved; only the latest document is right.
- temporal: "who led X on <date>" and "which tariff applied on <date>", answered by bracketing dated documents.
- multi_hop: 2- and 3-hop chains whose bridges (maker, supplier, acquirer, founder) are never named in the question.
- aggregation: totals and differences over quarterly reports, two to eight documents per question (a calculator helps).
- distractor: near-name companies and product codes whose facts differ.
- long_context: facts buried deep in ~4,000-word annual reports whose paragraphs rarely name the company.
- unanswerable (~25%): near misses: an attribute a sheet does not list, a period with no document, a false premise,
  an entity that does not exist, a fact the corpus never states.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "hard"
SEED = 20261025

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
ORDINALS = ["first", "second", "third", "fourth"]
CITIES = ["Port Avalen", "Kessrick", "Dunmore Reach", "Valtessa", "Ostrow Bay", "Halbrook", "Merrowgate", "Tirlan", "Saltmere", "Corvisk", "Brennock", "Ferrisvale"]
UNIVERSITIES = [
    "the University of Kessrick",
    "the Valtessa Institute of Technology",
    "Saltmere Maritime College",
    "Corvisk Polytechnic",
    "the Halbrook School of Engineering",
    "Merrowgate University",
]
FIELDS = ["mechanical engineering", "naval architecture", "physics", "electrical engineering", "economics", "materials science", "computer science"]
FIRST_NAMES = (
    "Ilse Marten Odile Pavel Rhea Soren Talia Ubaldo Vesna Wendell Ximena Yusuf Zora Anselm Brigid Casimir Delphine Emrys Fenna Gideon "
    "Hester Ivo Jolien Katarin Leopold Mireille Nikolai Oriane Piet Quilla Rosalind Stellan Theodora Ulric Vivienne Wilhelmina Xavier "
    "Yolanda Zeno Agathe Bastian Cordelia Dorian Elodie Florian Greta Hamish Isadora Jasper Kaia Lorcan Maren Nils Ottilie Percival Ragna "
    "Silas Tamsin Uriel Verena Annika Bertil Clemence Desmond"
).split()
LAST_NAMES = (
    "Arkwright Bellamy Castellane Drummond Eskeland Falkner Grisholm Halloway Ingersen Jessop Kilbride Lowenthal Marchetti Norquist Okafor "
    "Pemberton Quennell Rasmussen Sorrell Thackeray Underhill Varga Whitlock Yardley Zelenko Abernathy Blackwood Corrigan Dunleavy Ellingham "
    "Fairweather Gallagher Holmqvist Ivanova Jardine Kerrigan Lindgren Mortensen Nakamura Ostergaard Penhaligon Quarrie Ravensworth Stranack "
    "Tolliver Ulvaeus Vandermeer Wexford Ysselstein Zabriskie Ashdown Birchall Calloway Darrow Esposito Fontaine Gunnarsson Hartigan Imrie "
    "Jovanovic Kowalczyk Lysander Montague Nyberg"
).split()

# (name, short name, what the company does). The first sixteen keep sector words in their names; the four neutral names are the
# second supplier of a component type, so a chain question cannot be answered by guessing the supplier from its name.
CORE = [
    ("Brightwater Instruments", "Brightwater", "maker of marine survey instruments"),
    ("Calder Freight", "Calder", "coastal freight operator"),
    ("Ironvale Sensors", "Ironvale", "maker of pressure sensors"),
    ("Lumen Harbor Systems", "Lumen Harbor", "developer of port management software"),
    ("Norrland Batteries", "Norrland", "maker of marine battery packs"),
    ("Pellucid Optics", "Pellucid", "maker of camera lens assemblies"),
    ("Quarry Point Robotics", "Quarry Point", "maker of dock and inspection robots"),
    ("Redfern Composites", "Redfern", "maker of composite hull shells"),
    ("Sablewood Logistics", "Sablewood", "warehousing and logistics group"),
    ("Tessaline Marine", "Tessaline", "builder of small workboats"),
    ("Umbra Grid", "Umbra", "operator of harbour power grids"),
    ("Vantor Aerial", "Vantor", "maker of survey and inspection drones"),
    ("Westmark Polymers", "Westmark", "maker of polymer housings and hull parts"),
    ("Yarrow Drives", "Yarrow", "maker of electric drive motors"),
    ("Zephyrine Cables", "Zephyrine", "maker of subsea tether cables"),
    ("Ashgrove Analytics", "Ashgrove", "provider of fleet analytics software"),
    ("Kestova Energy", "Kestova", "maker of battery packs and chargers"),
    ("Marisco Devices", "Marisco", "maker of sensors and tether cables"),
    ("Orlan Glassworks", "Orlan", "maker of optical glass and lens assemblies"),
    ("Hessle Motorworks", "Hessle", "maker of electric drive motors"),
]
# (decoy name, the company it is easily confused with, what the decoy does)
DECOYS = [
    ("Brightwater Instrumentation", "Brightwater Instruments", "distributor of laboratory instruments"),
    ("Ironvale Sensing", "Ironvale Sensors", "maker of industrial gas detectors"),
    ("Norrland Battery Works", "Norrland Batteries", "recycler of industrial batteries"),
    ("Pellucid Optical", "Pellucid Optics", "chain of optician shops"),
    ("Tessaline Maritime", "Tessaline Marine", "ship-management agency"),
    ("Yarrow Drive Systems", "Yarrow Drives", "maker of conveyor drive systems"),
]
SUPPLIERS = {
    "battery pack": ["Norrland Batteries", "Kestova Energy"],
    "pressure sensor": ["Ironvale Sensors", "Marisco Devices"],
    "lens assembly": ["Pellucid Optics", "Orlan Glassworks"],
    "hull shell": ["Redfern Composites", "Westmark Polymers"],
    "drive motor": ["Yarrow Drives", "Hessle Motorworks"],
    "tether cable": ["Zephyrine Cables", "Marisco Devices"],
}
KINDS = {  # kind -> (components, special row label, unit, range)
    "drone": (("battery pack", "lens assembly"), "Maximum payload", "kg", (2, 14)),
    "boat": (("hull shell", "drive motor"), "Top speed", "knots", (18, 42)),
    "instrument": (("pressure sensor", "tether cable"), "Depth rating", "m", (150, 3000)),
    "robot": (("drive motor", "battery pack"), "Rated pull", "kN", (4, 40)),
}
KIND_PHRASES = {
    "drone": ["survey drone", "inspection drone"],
    "boat": ["workboat", "patrol boat"],
    "instrument": ["echo sounder", "sensor buoy"],
    "robot": ["mooring robot", "hull-cleaning crawler"],
}
REVISED_PRODUCTS = [  # three revisions each
    ("Harrier V8", "drone", "Vantor Aerial"),
    ("Kestrel K3", "drone", "Vantor Aerial"),
    ("Mako 620", "boat", "Tessaline Marine"),
    ("Orca 900", "boat", "Tessaline Marine"),
    ("Sounder S12", "instrument", "Brightwater Instruments"),
    ("Gullwing GW-2", "instrument", "Brightwater Instruments"),
    ("Dockhand R5", "robot", "Quarry Point Robotics"),
    ("Strider M4", "robot", "Quarry Point Robotics"),
]
PRODUCT_PAIRS = [  # near-identical codes from different makers
    (("Tidewatch TX-40", "instrument", "Brightwater Instruments"), ("Tidewatch TX-40S", "instrument", "Quarry Point Robotics")),
    (("Petrel P2", "drone", "Vantor Aerial"), ("Petrel P2X", "drone", "Quarry Point Robotics")),
    (("Lantern L7", "instrument", "Brightwater Instruments"), ("Lantern L7 Pro", "instrument", "Tessaline Marine")),
    (("Skimmer SK-10", "boat", "Tessaline Marine"), ("Skimmer SK-10M", "boat", "Quarry Point Robotics")),
]
SINGLE_PRODUCTS = [
    ("Heron H1", "drone", "Vantor Aerial"),
    ("Bollard B9", "robot", "Quarry Point Robotics"),
    ("Wavecrest W3", "boat", "Tessaline Marine"),
    ("Fathom F5", "instrument", "Brightwater Instruments"),
    ("Cormorant C2", "drone", "Vantor Aerial"),
    ("Tugmate T6", "robot", "Quarry Point Robotics"),
    ("Seaglass G4", "instrument", "Brightwater Instruments"),
    ("Puffin PF-1", "drone", "Vantor Aerial"),
]
MISSING_PRODUCTS = ["Harrier V9", "Tidewatch TX-50", "Kestrel K4", "Bollard B10"]
MISSING_COMPANIES = [("Pellucid Optronics", "In which year was Pellucid Optronics founded?"), ("Calder Freightlines", "Who founded Calder Freightlines?"),
                     ("Vantor Aeronautics", "Where is Vantor Aeronautics headquartered?")]
ACQUISITIONS = [  # (acquirer, target)
    ("Umbra Grid", "Norrland Batteries"),
    ("Ashgrove Analytics", "Ironvale Sensors"),
    ("Sablewood Logistics", "Zephyrine Cables"),
    ("Lumen Harbor Systems", "Pellucid Optics"),
    ("Calder Freight", "Redfern Composites"),
]
RELOCATED = ["Calder Freight", "Lumen Harbor Systems", "Sablewood Logistics", "Umbra Grid"]
SHIPPERS = ["Norrland Batteries", "Vantor Aerial", "Yarrow Drives"]
REGIONS = ["North Coast", "Inland", "Islands", "Export"]
ANNUAL_REPORTERS = ["Norrland Batteries", "Vantor Aerial", "Yarrow Drives", "Tessaline Marine"]
RESEARCH_CENTRES = ["Halden Research Centre", "Brisk Point Research Centre", "Saltglass Research Centre", "Mereworth Research Centre",
                    "Calloway Yard Research Centre", "Fenwick Research Centre", "Old Mill Research Centre", "Gannet Head Research Centre"]
REPORT_YEARS = (2024, 2025)  # questions ask about 2025; the 2024 report is the near-miss next to it
TARIFF_CHARGES = [("berth", "Berth fee, per metre of length per day", "berth fee per metre of length per day"),
                  ("pilot", "Pilotage, per movement", "pilotage charge per movement"),
                  ("box", "Container handling, per TEU", "container handling charge per TEU")]
PORTS = {"Port Avalen": [date(2021, 1, 1), date(2022, 7, 1), date(2024, 4, 1), date(2025, 3, 15)],
         "Ostrow Bay": [date(2021, 6, 1), date(2023, 1, 1), date(2024, 9, 1), date(2025, 7, 1)]}

TEAMS = ["the field service team", "the logistics desk", "the quality group", "the procurement office", "the training unit", "the customer support desk",
         "the engineering office", "the facilities team"]
TOPICS = ["spare-parts planning", "supplier audits", "workshop safety", "digital service records", "energy metering", "customer onboarding",
          "warehouse layout", "returns handling", "calibration routines", "document control", "shift planning", "packaging waste"]
SITES = ["the main workshop", "the northern depot", "the coastal test site", "the assembly hall", "the service centre", "the training rooms"]
FILLER = [
    "{Org} reviewed its approach to {topic} during the year and simplified several internal steps.",
    "In {month}, {team} completed a review of {topic} at {site}.",
    "{Org} ran {n} workshops with customers to collect feedback on service quality.",
    "Work on {topic} continued, with most of the planned changes now in place.",
    "{Team} introduced a shorter checklist for {topic}, which reduced rework.",
    "{Org} continued to invest in {site}, replacing older equipment where it was no longer economical to repair.",
    "Demand from port operators remained steady, while survey and inspection customers ordered more often but in smaller volumes.",
    "The board discussed {topic} at {n} of its meetings.",
    "{Org} kept its customer satisfaction survey unchanged so that results can be compared over time.",
    "A pilot of {topic} at {site} ran from {month} to {month2} and will be extended.",
    "{Team} took on {n} new colleagues, most of them through internal transfers.",
    "Lead times for standard orders shortened slightly over the second half.",
    "{Org} renegotiated several supply agreements to include clearer delivery terms.",
    "Freight costs were broadly stable, although winter storms delayed a number of coastal deliveries.",
    "{Org} updated its code of conduct and asked every employee to confirm they had read it.",
    "Internal audit looked at {topic} and reported no significant findings.",
    "The management team met customers at {n} regional trade events.",
    "Spare-parts availability improved after {team} reorganised {site}.",
    "{Org} expects the market for coastal monitoring to keep growing at a moderate pace.",
    "Energy use per unit produced fell after the lighting at {site} was replaced.",
    "{Org} continued to support local schools with workshop visits and summer placements.",
    "Several long-standing customers renewed their service agreements on similar terms.",
    "Changes to {topic} were rolled out in stages to limit disruption.",
    "{Team} reported {n} minor incidents, none of which caused lost working time.",
    "The finance team moved its monthly reporting to a shared platform.",
    "{Org} published a revised supplier handbook in {month}.",
    "A new scheduling tool now plans work at {site} a week ahead.",
    "Customer complaints were handled within the target time in most cases.",
    "{Org} reviewed its insurance cover and kept the main policies with existing providers.",
    "Training hours per employee rose, mainly because of new safety modules.",
    "{Org} sees exchange-rate movements and coastal weather as the main sources of uncertainty.",
    "Investment plans focus on {topic} and on {topic2}.",
    "The quality group repeated its check of {topic} and closed all open actions.",
    "{Org} joined an industry working group on shared standards for {topic}.",
    "Maintenance at {site} was moved to quieter weeks to avoid peak demand.",
    "Feedback from the employee survey was shared with every team in {month}.",
]
PROFILE_FILLER = [
    "{short} describes its customers as port operators, survey firms and public agencies.",
    "Most of its revenue comes from long-term service agreements rather than one-off sales.",
    "The register lists {short} as a private limited company.",
    "The company publishes a short sustainability statement each spring.",
    "Its staff are organised into engineering, operations and customer support teams.",
    "{short} takes part in the Meridian Coast apprenticeship scheme.",
    "The company holds a quality certificate for its main workshop.",
    "It files its annual return with the register on time.",
]
REPORT_SECTIONS = ["Chair's letter", "The year in brief", "Markets", "Operations", "People", "Research and development", "Sustainability", "Governance",
                   "Risks and outlook"]


def fmt_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def article(phrase: str) -> str:
    return f"an {phrase}" if phrase[0] in "aeiou" else f"a {phrase}"


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def money(value: int) -> str:
    return f"MC {value:,}"


@dataclass
class Company:
    name: str
    short: str
    sector: str
    founded: int
    city: str
    founder: str
    decoy: bool = False
    successors: list[tuple[str, date, date]] = field(default_factory=list)  # (person, announced, effective)
    moved: tuple[str, date] | None = None
    acquired: tuple[str, date] | None = None


@dataclass
class Product:
    name: str
    kind: str
    phrase: str
    maker: str
    suppliers: dict[str, str]
    revisions: list[dict[str, Any]]  # one row set per revision; a product without revisions has one
    weight: float | None


@dataclass
class Doc:
    title: str
    blocks: list[str]

    def text(self) -> str:
        return f"# {self.title}\n\n" + "\n\n".join(self.blocks) + "\n"


class Suite:
    def __init__(self, seed: int = SEED) -> None:
        self.rng = random.Random(seed)
        self.docs: dict[str, Doc] = {}
        self.ids: dict[str, str] = {}
        self.questions: list[dict[str, Any]] = []
        self.facts_annual: dict[tuple[str, int], dict[str, str]] = {}
        firsts, lasts = FIRST_NAMES[:], LAST_NAMES[:]
        self.rng.shuffle(firsts)
        self.rng.shuffle(lasts)
        self._people = [f"{first} {last}" for first, last in zip(firsts, lasts, strict=True)]

    # ----- world -----------------------------------------------------------------------------------------------------------------
    def person(self) -> str:
        return self._people.pop()

    def day_between(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, (end - start).days))

    def build_world(self) -> None:
        rng = self.rng
        years = rng.sample(range(1986, 2016), len(CORE) + len(DECOYS))
        self.companies: dict[str, Company] = {}
        for (name, short, sector), year in zip(CORE, years, strict=False):
            self.companies[name] = Company(name, short, sector, year, rng.choice(CITIES), self.person())
        for (name, twin, sector), year in zip(DECOYS, years[len(CORE) :], strict=True):
            city = rng.choice([c for c in CITIES if c != self.companies[twin].city])
            self.companies[name] = Company(name, name, sector, year, city, self.person(), decoy=True)
        for company in self.core():
            count = rng.choice([1, 1, 2])
            starts = sorted(self.day_between(date(2019, 3, 1), date(2025, 4, 30)) for _ in range(count))
            if count == 2 and (starts[1] - starts[0]).days < 400:
                starts[1] = min(starts[0] + timedelta(days=400 + rng.randint(0, 200)), date(2025, 6, 30))
            for effective in starts:
                company.successors.append((self.person(), effective - timedelta(days=rng.randint(14, 45)), effective))
        for name in RELOCATED:
            company = self.companies[name]
            used = {c.city for c in self.companies.values() if c.name != name}
            new_city = rng.choice([c for c in CITIES if c != company.city and c not in used] or [c for c in CITIES if c != company.city])
            company.moved = (new_city, self.day_between(date(2023, 6, 1), date(2025, 6, 30)))
        for acquirer, target in ACQUISITIONS:
            self.companies[target].acquired = (acquirer, self.day_between(date(2023, 3, 1), date(2025, 8, 31)))
        self.bios = {company.founder: (rng.choice(CITIES), company.founded - rng.randint(28, 45), rng.choice(UNIVERSITIES), rng.choice(FIELDS))
                     for company in self.core()}

        self.products: dict[str, Product] = {}
        singles = [p for pair in PRODUCT_PAIRS for p in pair] + SINGLE_PRODUCTS
        for name, kind, maker in REVISED_PRODUCTS + singles:
            components, _, _, (low, high) = KINDS[kind]
            suppliers = {component: rng.choice(SUPPLIERS[component]) for component in components}
            price = rng.randrange(1800, 24000, 10)
            warranty = rng.choice([12, 24])
            endurance = rng.randint(30, 140) / 10
            special = rng.randint(low, high)
            rows = [{"price": price, "warranty": warranty, "endurance": endurance, "special": special}]
            if (name, kind, maker) in REVISED_PRODUCTS:
                rows.append({**rows[0], "price": price + rng.randrange(150, 900, 10), "endurance": round(endurance + rng.choice([-0.5, 0.5, 1.0]), 1)})
                rows.append({**rows[1], "price": rows[1]["price"] + rng.randrange(150, 900, 10), "warranty": warranty + 12})
            weight = round(rng.uniform(3, 60), 1) if rng.random() < 0.55 else None
            self.products[name] = Product(name, kind, rng.choice(KIND_PHRASES[kind]), maker, suppliers, rows, weight)
        self.revision_dates = {name: [self.day_between(date(2023, 1, 10), date(2023, 6, 30)), self.day_between(date(2024, 2, 1), date(2024, 7, 31)),
                                      self.day_between(date(2025, 3, 1), date(2025, 9, 30))] for name, _, _ in REVISED_PRODUCTS}

        self.shipments = {(name, year, q): {region: rng.randint(400, 2600) for region in REGIONS}
                          for name in SHIPPERS for year in (2024, 2025) for q in range(1, 5)}
        self.tariffs = {}
        for port, dates in PORTS.items():
            berth, pilot, box = rng.randint(250, 320), rng.randint(380, 460), rng.randint(48, 62)
            editions = []
            for _ in dates:
                editions.append({"berth": f"MC {berth / 100:.2f}", "pilot": money(pilot), "box": money(box)})
                berth, pilot, box = berth + rng.randint(10, 40), pilot + rng.randint(15, 45), box + rng.randint(2, 6)
            self.tariffs[port] = editions
        centres = RESEARCH_CENTRES[:]
        rng.shuffle(centres)
        self.annual = {(name, year): {"employees": rng.randint(310, 2900), "centre": centres.pop(), "centre_city": rng.choice(CITIES),
                                      "patents": rng.randint(11, 64), "audit_chair": self.person()} for name in ANNUAL_REPORTERS for year in REPORT_YEARS}

    def core(self) -> list[Company]:
        return [c for c in self.companies.values() if not c.decoy]

    def ceo_on(self, company: Company, day: date) -> str:
        current = company.founder
        for person, _, effective in company.successors:
            if effective <= day:
                current = person
        return current

    # ----- text ------------------------------------------------------------------------------------------------------------------
    def filler(self, count: int, org: str = "the company", named: float = 0.2, pool: list[str] = FILLER) -> list[str]:
        rng = self.rng
        sentences = []
        for template in (rng.choice(pool) for _ in range(count)):
            ref = org if rng.random() < named else rng.choice(["the company", "the group"])
            month, month2 = sorted(rng.sample(range(12), 2))
            topic, topic2 = rng.sample(TOPICS, 2)
            team = rng.choice(TEAMS)
            sentences.append(template.format(Org=ref[0].upper() + ref[1:], org=ref, short=org, topic=topic, topic2=topic2, team=team,
                                             Team=team[0].upper() + team[1:], site=rng.choice(SITES), n=rng.randint(3, 40),
                                             month=MONTHS[month], month2=MONTHS[month2]))
        return sentences

    @staticmethod
    def paragraphs(sentences: list[str], size: int = 5) -> list[str]:
        return [" ".join(sentences[i : i + size]) for i in range(0, len(sentences), size)]

    def add(self, key: str, title: str, blocks: list[str]) -> None:
        assert key not in self.docs, key
        self.docs[key] = Doc(title, blocks)

    # ----- documents -------------------------------------------------------------------------------------------------------------
    def founded_sentence(self, c: Company) -> str:
        return f"It was founded in {c.founded} in {c.city}, where its head office is located."

    def founder_sentence(self, c: Company) -> str:
        return f"{c.founder} founded the business." if c.decoy else f"{c.founder} founded {c.short} and served as its first chief executive."

    def appointment_sentence(self, c: Company, person: str, announced: date, effective: date) -> str:
        return f"{c.name} announced on {fmt_date(announced)} that its board had appointed {person} as chief executive officer with effect from {fmt_date(effective)}."

    def write_documents(self) -> None:
        rng = self.rng
        others = [c.name for c in self.core()]
        for c in self.companies.values():
            review = self.day_between(date(2022, 1, 15), date(2022, 11, 30))
            lead = f"*Profile prepared by the Meridian Coast Business Register, last reviewed {fmt_date(review)}.*"
            body = [f"{c.name} is {article(c.sector)}.", self.founded_sentence(c), self.founder_sentence(c)]
            body += rng.sample(PROFILE_FILLER, 4)
            self.add(f"profile:{c.name}", f"{c.name}: company profile", [lead, " ".join(s.replace("{short}", c.short) for s in body)])
        for c in self.core():
            previous = c.founder
            for i, (person, announced, effective) in enumerate(c.successors):
                leaving = rng.choice(["is retiring", "will join the board as a non-executive director", "is leaving to pursue other interests"])
                role = rng.choice(["chief operating officer", "finance director", "head of engineering", "managing director of the export business"])
                former = rng.choice([o for o in others if o != c.name])
                text = [self.appointment_sentence(c, person, announced, effective), f"{person} succeeds {previous}, who {leaving}.",
                        f"{person} was previously {role} at {former}."] + self.filler(2, c.short, named=0.5)
                self.add(f"ceo:{c.name}:{i}", f"{c.name} appoints {person} as chief executive", self.paragraphs(text, 3))
                previous = person
            if c.moved:
                city, day = c.moved
                text = [f"{c.name} completed the move of its head office from {c.city} to {city} on {fmt_date(day)}.",
                        f"The former {c.city} office will stay open as a regional service depot."] + self.filler(3, c.short, named=0.5)
                self.add(f"move:{c.name}", f"{c.name} moves its head office", self.paragraphs(text, 3))
            if c.acquired:
                acquirer, day = c.acquired
                text = [f"{acquirer} completed its acquisition of {c.name} on {fmt_date(day)}, after approval by the Meridian Coast Competition Office.",
                        f"{c.name} keeps its name and its own management team.", "The parties did not disclose the price."] + self.filler(2, acquirer, named=0.5)
                self.add(f"acq:{c.name}", f"{acquirer} completes acquisition of {c.name}", self.paragraphs(text, 3))
            city, born, university, subject = self.bios[c.founder]
            text = [f"{c.founder} was born in {city} in {born}.", f"{c.founder} studied {subject} at {university}.",
                    f"{c.founder} founded {c.name} in {c.founded}.",
                    rng.choice(["Outside work, the founder sails and coaches a junior rowing crew.", "The founder sits on the board of a regional maritime museum.",
                                "The founder mentors engineering students through a local scheme."])]
            self.add(f"bio:{c.founder}", f"{c.founder}: biography", [" ".join(text)])

        for p in self.products.values():
            _, label, unit, _ = KINDS[p.kind]
            for r, rows in enumerate(p.revisions):
                table = ["| Item | Value |", "| --- | --- |", f"| List price | {money(rows['price'])} |", f"| Warranty | {rows['warranty']} months |",
                         f"| Endurance | {rows['endurance']} hours |", f"| {label} | {rows['special']:,} {unit} |"]
                if p.weight is not None:
                    table.append(f"| Weight | {p.weight} kg |")
                lead = [self.maker_sentence(p)]
                if len(p.revisions) > 1:
                    letter = "ABC"[r]
                    lead.insert(0, f"Revision {letter}, effective {fmt_date(self.revision_dates[p.name][r])}." + (f" This revision supersedes revision {'ABC'[r - 1]}." if r else ""))
                components = " ".join(self.supply_sentence(p, component) for component in p.suppliers)
                title = f"{p.name} specification sheet" + (f", revision {'ABC'[r]}" if len(p.revisions) > 1 else "")
                self.add(self.spec_key(p, r), title, [" ".join(lead), "\n".join(table), components])

        for (name, year, q), regions in self.shipments.items():
            c = self.companies[name]
            table = ["| Region | Units shipped |", "| --- | --- |"] + [f"| {region} | {units:,} |" for region, units in regions.items()]
            table.append(f"| Total | {sum(regions.values()):,} |")
            intro = f"This report covers the operations of {name} in the {ORDINALS[q - 1]} quarter of {year}."
            self.add(f"ship:{name}:{year}:{q}", f"{name} quarterly operations report, Q{q} {year}",
                     [intro, *self.paragraphs(self.filler(7, c.short)), "\n".join(table), *self.paragraphs(self.filler(6, c.short))])

        for port, dates in PORTS.items():
            for e, (effective, rates) in enumerate(zip(dates, self.tariffs[port], strict=True)):
                lead = f"Issued by the {port} Harbour Board. This edition takes effect on {fmt_date(effective)}" + (
                    f" and replaces the edition of {fmt_date(dates[e - 1])}." if e else " and is the first consolidated edition.")
                table = ["| Charge | Rate |", "| --- | --- |", *(f"| {label} | {rates[key]} |" for key, label, _ in TARIFF_CHARGES)]
                if port == "Port Avalen":
                    table.append(f"| Waste reception, per call | {money(120 + 10 * e)} |")
                notes = ["Charges are invoiced to the ship's agent after departure.", "Vessels on scheduled ferry services pay under a separate agreement.",
                         "Disputes about an invoice must be raised within thirty days."]
                self.add(f"tariff:{port}:{e}", f"{port} harbour tariff, {effective.year} edition", [lead, "\n".join(table), " ".join(notes)])

        for (name, year), facts in self.annual.items():
            c = self.companies[name]
            fact_sentences = {
                "People": f"At the close of the year {'the group' if rng.random() < 0.5 else c.short} employed {facts['employees']:,} people across all sites.",
                "Research and development": f"In {MONTHS[rng.randint(2, 10)]} the group opened the {facts['centre']} in {facts['centre_city']}.",
                "Governance": f"{facts['audit_chair']} chairs the audit committee.",
            }
            patents = f"During the year {'the company' if rng.random() < 0.5 else c.short} filed {facts['patents']} patent applications."
            blocks = [f"This annual report of {name} covers the year ended 31 December {year}."]
            for section in REPORT_SECTIONS:
                sentences = self.filler(28, c.short, named=0.12)
                if section in fact_sentences:
                    sentences.insert(rng.randint(6, 20), fact_sentences[section])
                if section == "Research and development":
                    sentences.insert(rng.randint(22, 27), patents)
                blocks += [f"## {section}", *self.paragraphs(sentences, 6)]
            self.facts_annual[(name, year)] = {**fact_sentences, "patents": patents}
            self.add(f"annual:{name}:{year}", f"{name} annual report {year}", blocks)

    def maker_sentence(self, p: Product) -> str:
        return f"The {p.name} is {article(p.phrase)} designed and built by {p.maker}."

    def supply_sentence(self, p: Product, component: str) -> str:
        return f"The {component} is supplied by {p.suppliers[component]}."

    @staticmethod
    def spec_key(p: Product, revision: int = -1) -> str:
        return f"spec:{p.name}:{revision % len(p.revisions)}"

    def assign_ids(self) -> None:
        keys = list(self.docs)
        self.rng.shuffle(keys)
        self.ids = {key: f"doc_{i:03d}" for i, key in enumerate(keys, start=1)}

    # ----- questions -------------------------------------------------------------------------------------------------------------
    def ask(self, category: str, question: str, reference: str, keywords: list[str], relevant: list[tuple[str, int]], evidence: list[tuple[str, str]],
            difficulty: str = "hard", answer_type: str = "single_fact", tools: list[str] | None = None, routing: str | None = None,
            **metadata: Any) -> None:
        for key, quote in evidence:
            assert quote in self.docs[key].text(), (key, quote)
        record: dict[str, Any] = {"question": question, "reference_answer": reference, "expected_keywords": keywords,
                                  "relevant_doc_ids": [self.ids[key] for key, _ in relevant], "category": category, "difficulty": difficulty,
                                  "answer_type": answer_type}
        if tools:
            record["requires_tools"] = tools
        if routing:
            record["routing_hint"] = routing
        if category == "unanswerable":
            record["answerable"] = False
        if evidence:
            record["evidence"] = [{"doc_id": self.ids[key], "quote": quote} for key, quote in evidence]
        if metadata:
            record["metadata"] = metadata
        record["_qrels"] = [(self.ids[key], grade) for key, grade in relevant]
        self.questions.append(record)

    def unanswerable(self, trap: str, question: str, answer: str) -> None:
        self.ask("unanswerable", question, answer, [], [], [], difficulty="medium", answer_type="unanswerable", trap=trap)

    def write_questions(self) -> None:
        self.version_questions()
        self.temporal_questions()
        self.multi_hop_questions()
        self.aggregation_questions()
        self.distractor_questions()
        self.long_context_questions()
        self.unanswerable_questions()

    def version_questions(self) -> None:
        for name, _, _ in REVISED_PRODUCTS:
            p = self.products[name]
            latest = p.revisions[-1]
            superseded = [self.ids[self.spec_key(p, r)] for r in range(len(p.revisions) - 1)]
            price = money(latest["price"])
            self.ask("version_conflict", f"What is the current list price of the {name}?", f"The current list price of the {name} is {price} (revision C).",
                     [price], [(self.spec_key(p), 3)], [(self.spec_key(p), f"| List price | {price} |")], difficulty="medium", superseded=superseded,
                     answer=price)
            warranty = f"{latest['warranty']} months"
            self.ask("version_conflict", f"How long is the warranty on the {name} now?", f"The {name} now carries a {warranty} warranty (revision C).",
                     [warranty], [(self.spec_key(p), 3)], [(self.spec_key(p), f"| Warranty | {warranty} |")], difficulty="medium", superseded=superseded,
                     answer=warranty)
        for name in RELOCATED:
            c = self.companies[name]
            city, _ = c.moved  # type: ignore[misc]
            sentence = self.docs[f"move:{name}"].blocks[0].split(". ")[0] + "."
            self.ask("version_conflict", f"In which city is {possessive(name)} head office today?", f"{possessive(name)} head office is in {city}; it moved there from {c.city}.",
                     [city], [(f"move:{name}", 3)], [(f"move:{name}", sentence)], difficulty="medium", superseded=[self.ids[f"profile:{name}"]], answer=city)

    def temporal_questions(self) -> None:
        rng = self.rng
        for c in rng.sample(self.core(), 10):
            starts = [date(2018, 1, 1)] + [effective for _, _, effective in c.successors]
            ends = [announced for _, announced, _ in c.successors] + [date(2025, 12, 31)]
            i = rng.randrange(len(starts))
            day = self.day_between(starts[i] + timedelta(days=60), ends[i] - timedelta(days=60))
            person = self.ceo_on(c, day)
            if i == 0:
                relevant, evidence = [(f"profile:{c.name}", 3)], [(f"profile:{c.name}", self.founder_sentence(c))]
            else:
                person_, announced, effective = c.successors[i - 1]
                relevant, evidence = [(f"ceo:{c.name}:{i - 1}", 3)], [(f"ceo:{c.name}:{i - 1}", self.appointment_sentence(c, person_, announced, effective))]
            if i < len(c.successors):
                relevant.append((f"ceo:{c.name}:{i}", 2))
            self.ask("temporal", f"Who was the chief executive of {c.name} on {fmt_date(day)}?", f"{person} was chief executive of {c.name} on {fmt_date(day)}.",
                     [person], relevant, evidence)
        for port, dates in PORTS.items():
            for e in range(len(dates)):
                end = dates[e + 1] - timedelta(days=30) if e + 1 < len(dates) else date(2025, 12, 31)
                day = self.day_between(dates[e] + timedelta(days=30), end)
                key, label, phrase = rng.choice(TARIFF_CHARGES)
                rate = self.tariffs[port][e][key]
                relevant = [(f"tariff:{port}:{e}", 3)] + ([(f"tariff:{port}:{e + 1}", 2)] if e + 1 < len(dates) else [])
                self.ask("temporal", f"Under the {port} harbour tariff, what was the {phrase} for a ship arriving on {fmt_date(day)}?",
                         f"The {dates[e].year} edition applied, so the {phrase} was {rate}.", [rate], relevant,
                         [(f"tariff:{port}:{e}", f"| {label} | {rate} |")], difficulty="medium")

    def multi_hop_questions(self) -> None:
        rng = self.rng
        singles = [self.products[name] for pair in PRODUCT_PAIRS for name, _, _ in pair] + [self.products[name] for name, _, _ in SINGLE_PRODUCTS]
        order = singles[:]
        rng.shuffle(order)

        def maker_chain(p: Product) -> tuple[Company, list[tuple[str, int]], list[tuple[str, str]]]:
            maker = self.companies[p.maker]
            return maker, [(self.spec_key(p), 3), (f"profile:{maker.name}", 3)], [(self.spec_key(p), self.maker_sentence(p)), (f"profile:{maker.name}", self.founded_sentence(maker))]

        for p in order[:6]:  # product -> maker -> founding city
            maker, relevant, evidence = maker_chain(p)
            self.ask("multi_hop", f"In which city was the company that builds the {p.name} founded?", f"The {p.name} is built by {maker.name}, which was founded in {maker.city}.",
                     [maker.city], relevant, evidence, difficulty="medium", routing="multi_hop", hops=2, bridges=[maker.name])
        for p in order[6:10]:  # product -> supplier -> founding year
            component = rng.choice(list(p.suppliers))
            supplier = self.companies[p.suppliers[component]]
            self.ask("multi_hop", f"In what year was the company that supplies the {component} of the {p.name} founded?",
                     f"{supplier.name} supplies the {component} of the {p.name}; it was founded in {supplier.founded}.", [str(supplier.founded)],
                     [(self.spec_key(p), 3), (f"profile:{supplier.name}", 3)],
                     [(self.spec_key(p), self.supply_sentence(p, component)), (f"profile:{supplier.name}", self.founded_sentence(supplier))],
                     difficulty="medium", routing="multi_hop", hops=2, bridges=[supplier.name])
        for acquirer_name, target_name in ACQUISITIONS:  # target -> acquirer -> founder
            acquirer, target = self.companies[acquirer_name], self.companies[target_name]
            sentence = self.docs[f"acq:{target.name}"].blocks[0].split(". ")[0]
            sentence = sentence if sentence.endswith(".") else sentence + "."
            self.ask("multi_hop", f"Who founded the company that acquired {target.name}?", f"{acquirer.name} acquired {target.name}; {acquirer.name} was founded by {acquirer.founder}.",
                     [acquirer.founder], [(f"acq:{target.name}", 3), (f"profile:{acquirer.name}", 3)],
                     [(f"acq:{target.name}", sentence), (f"profile:{acquirer.name}", self.founder_sentence(acquirer))],
                     difficulty="medium", routing="multi_hop", hops=2, bridges=[acquirer.name])
        for c in rng.sample([c for c in self.core() if c.name not in ANNUAL_REPORTERS], 3):  # company -> founder -> university
            _, _, university, _ = self.bios[c.founder]
            self.ask("multi_hop", f"Which university did the founder of {c.name} attend?", f"{c.founder} founded {c.name} and studied at {university}.",
                     [university.removeprefix("the ")], [(f"profile:{c.name}", 3), (f"bio:{c.founder}", 3)],
                     [(f"profile:{c.name}", self.founder_sentence(c)), (f"bio:{c.founder}", self.bio_sentence(c.founder, "studied"))],
                     difficulty="medium", routing="multi_hop", hops=2, bridges=[c.founder])
        for p in order[10:14]:  # product -> maker -> founder -> birthplace
            maker, relevant, evidence = maker_chain(p)
            city = self.bios[maker.founder][0]
            evidence = [evidence[0], (f"profile:{maker.name}", self.founder_sentence(maker)), (f"bio:{maker.founder}", self.bio_sentence(maker.founder, "born"))]
            self.ask("multi_hop", f"In which city was the founder of the company that builds the {p.name} born?",
                     f"The {p.name} is built by {maker.name}, founded by {maker.founder}, who was born in {city}.", [city],
                     [*relevant, (f"bio:{maker.founder}", 3)], evidence, routing="multi_hop", hops=3, bridges=[maker.name, maker.founder])
        acquired = {target for _, target in ACQUISITIONS}
        chains = [(p, comp) for p in order for comp, supplier in p.suppliers.items() if supplier in acquired]
        used: set[str] = set()
        picked = []
        for p, component in chains:
            if p.name not in used and len(picked) < 4:
                used.add(p.name)
                picked.append((p, component))
        assert len(picked) == 4, "not enough products with an acquired supplier: change the seed"
        for p, component in picked:  # product -> supplier -> acquirer -> founding year
            supplier = self.companies[p.suppliers[component]]
            acquirer = self.companies[supplier.acquired[0]]  # type: ignore[index]
            sentence = self.docs[f"acq:{supplier.name}"].blocks[0].split(". ")[0]
            sentence = sentence if sentence.endswith(".") else sentence + "."
            self.ask("multi_hop", f"In what year was the company that acquired the supplier of the {possessive(p.name)} {component} founded?",
                     f"{supplier.name} supplies the {component}; it was acquired by {acquirer.name}, founded in {acquirer.founded}.", [str(acquirer.founded)],
                     [(self.spec_key(p), 3), (f"acq:{supplier.name}", 3), (f"profile:{acquirer.name}", 3)],
                     [(self.spec_key(p), self.supply_sentence(p, component)), (f"acq:{supplier.name}", sentence), (f"profile:{acquirer.name}", self.founded_sentence(acquirer))],
                     routing="multi_hop", hops=3, bridges=[supplier.name, acquirer.name])
        rest = [p for p in order if p.name not in used][:4]
        for p in rest:  # product -> supplier -> founder -> university
            component = rng.choice(list(p.suppliers))
            supplier = self.companies[p.suppliers[component]]
            university = self.bios[supplier.founder][2]
            self.ask("multi_hop", f"Which university did the founder of the {component} supplier of the {p.name} attend?",
                     f"{supplier.name} supplies the {component}; its founder {supplier.founder} studied at {university}.", [university.removeprefix("the ")],
                     [(self.spec_key(p), 3), (f"profile:{supplier.name}", 3), (f"bio:{supplier.founder}", 3)],
                     [(self.spec_key(p), self.supply_sentence(p, component)), (f"profile:{supplier.name}", self.founder_sentence(supplier)),
                      (f"bio:{supplier.founder}", self.bio_sentence(supplier.founder, "studied"))],
                     routing="multi_hop", hops=3, bridges=[supplier.name, supplier.founder])

    def bio_sentence(self, person: str, kind: str) -> str:
        city, born, university, subject = self.bios[person]
        return f"{person} was born in {city} in {born}." if kind == "born" else f"{person} studied {subject} at {university}."

    def aggregation_questions(self) -> None:
        rng = self.rng

        def total(name: str, year: int, q: int) -> int:
            return sum(self.shipments[(name, year, q)].values())

        def total_row(name: str, year: int, q: int) -> tuple[str, str]:
            return f"ship:{name}:{year}:{q}", f"| Total | {total(name, year, q):,} |"

        def ask(question: str, answer: str, value: int, rows: list[tuple[str, str]], difficulty: str = "hard") -> None:
            keys = list(dict.fromkeys(key for key, _ in rows))
            self.ask("aggregation", question, answer, [f"{value:,}"], [(key, 3) for key in keys], rows, difficulty=difficulty, answer_type="number",
                     tools=["calculator"], routing="computation")

        for name in SHIPPERS:
            for year in (2024, 2025):
                values = [total(name, year, q) for q in range(1, 5)]
                ask(f"How many units did {name} ship in {year} in total?", f"{name} shipped {sum(values):,} units in {year} ({' + '.join(f'{v:,}' for v in values)}).",
                    sum(values), [total_row(name, year, q) for q in range(1, 5)])
        for name in SHIPPERS:
            region, year = rng.choice(REGIONS), rng.choice([2024, 2025])
            values = [self.shipments[(name, year, q)][region] for q in range(1, 5)]
            ask(f"How many units did {name} ship to the {region} region across the four quarters of {year}?",
                f"{name} shipped {sum(values):,} units to {region} in {year} ({' + '.join(f'{v:,}' for v in values)}).", sum(values),
                [(f"ship:{name}:{year}:{q}", f"| {region} | {self.shipments[(name, year, q)][region]:,} |") for q in range(1, 5)])
        for name in SHIPPERS:
            q = rng.randint(1, 4)
            before, after = total(name, 2024, q), total(name, 2025, q)
            change = "rose" if after > before else "fell"
            ask(f"By how many units did {possessive(name)} total shipments change from Q{q} 2024 to Q{q} 2025?",
                f"They {change} by {abs(after - before):,} units, from {before:,} to {after:,}.", abs(after - before),
                [total_row(name, 2024, q), total_row(name, 2025, q)], difficulty="medium")
        for name in SHIPPERS:
            year = rng.choice([2024, 2025])
            values = [total(name, year, q) for q in (1, 2)]
            ask(f"How many units did {name} ship in the first half of {year}?", f"{name} shipped {sum(values):,} units in the first half of {year} ({values[0]:,} + {values[1]:,}).",
                sum(values), [total_row(name, year, 1), total_row(name, year, 2)], difficulty="medium")
        for year in (2024, 2025):
            a, b = rng.sample(SHIPPERS, 2)
            ta, tb = (sum(total(n, year, q) for q in range(1, 5)) for n in (a, b))
            if ta < tb:
                a, b, ta, tb = b, a, tb, ta
            ask(f"How many more units did {a} ship than {b} in {year}?", f"{a} shipped {ta:,} units and {b} {tb:,}, a difference of {ta - tb:,}.", ta - tb,
                [total_row(a, year, q) for q in range(1, 5)] + [total_row(b, year, q) for q in range(1, 5)])

    def distractor_questions(self) -> None:
        for decoy_name, twin_name, _ in DECOYS:
            decoy, twin = self.companies[decoy_name], self.companies[twin_name]
            self.ask("distractor", f"In which year was {decoy.name} founded?", f"{decoy.name} was founded in {decoy.founded}.", [str(decoy.founded)],
                     [(f"profile:{decoy.name}", 3)], [(f"profile:{decoy.name}", self.founded_sentence(decoy))], difficulty="medium",
                     decoy=self.ids[f"profile:{twin.name}"], answer=str(decoy.founded))
            self.ask("distractor", f"Who founded {twin.name}?", f"{twin.founder} founded {twin.name}.", [twin.founder], [(f"profile:{twin.name}", 3)],
                     [(f"profile:{twin.name}", self.founder_sentence(twin))], difficulty="medium", decoy=self.ids[f"profile:{decoy.name}"], answer=twin.founder)
        for (base_name, _, _), (variant_name, _, _) in PRODUCT_PAIRS:
            base, variant = self.products[base_name], self.products[variant_name]
            price = money(variant.revisions[0]["price"])
            self.ask("distractor", f"What is the list price of the {variant.name}?", f"The {variant.name} lists at {price}.", [price], [(self.spec_key(variant), 3)],
                     [(self.spec_key(variant), f"| List price | {price} |")], difficulty="medium", decoy=self.ids[self.spec_key(base)], answer=price)

    def long_context_questions(self) -> None:
        for name in ANNUAL_REPORTERS:
            facts, sentences, key = self.annual[(name, 2025)], self.facts_annual[(name, 2025)], f"annual:{name}:2025"
            employees = f"{facts['employees']:,}"
            self.ask("long_context", f"How many people did {name} employ at the end of 2025?", f"{name} employed {employees} people at the end of 2025.",
                     [employees], [(key, 3)], [(key, sentences["People"])])
            self.ask("long_context", f"Which research centre did {name} open in 2025, and in which city?",
                     f"{name} opened the {facts['centre']} in {facts['centre_city']}.", [facts["centre"], facts["centre_city"]], [(key, 3)],
                     [(key, sentences["Research and development"])], answer_type="list")
            self.ask("long_context", f"How many patent applications did {name} file in 2025?", f"{name} filed {facts['patents']} patent applications in 2025.",
                     [str(facts["patents"])], [(key, 3)], [(key, sentences["patents"])])
            self.ask("long_context", f"Who chaired the audit committee of {name} in 2025?", f"{facts['audit_chair']} chaired {possessive(name)} audit committee in 2025.",
                     [facts["audit_chair"]], [(key, 3)], [(key, sentences["Governance"])])

    def unanswerable_questions(self) -> None:
        rng = self.rng
        no_weight = [p for p in self.products.values() if p.weight is None]
        for p in rng.sample(no_weight, 6):
            self.unanswerable("missing_attribute", f"How much does the {p.name} weigh?", f"The documents do not give a weight for the {p.name}.")
        self.unanswerable("missing_attribute", "What does Ostrow Bay charge for waste reception per call?",
                          "The Ostrow Bay tariff lists no waste reception charge.")
        for name in rng.sample(SHIPPERS, 3):
            q = rng.randint(1, 4)
            self.unanswerable("out_of_range", f"How many units did {name} ship in Q{q} 2023?", f"The documents have no report for {name} in Q{q} 2023.")
        for name in rng.sample(SHIPPERS, 2):
            self.unanswerable("out_of_range", f"How many units did {name} ship in the first quarter of 2026?", f"The documents have no report for {name} in 2026.")
        for port, year in (("Port Avalen", 2019), ("Ostrow Bay", 2020)):
            self.unanswerable("out_of_range", f"What pilotage charge did {port} apply to ships arriving in {year}?",
                              f"The {port} tariffs in the documents start in 2021; none covers {year}.")
        for acquirer_name, target_name in ACQUISITIONS[:4]:
            wrong = rng.choice([a for a, _ in ACQUISITIONS if a != acquirer_name])
            self.unanswerable("false_premise", f"When did {wrong} acquire {target_name}?",
                              f"The documents do not record {wrong} acquiring {target_name}; {acquirer_name} acquired it.")
        for c in rng.sample([c for c in self.core() if c.successors], 4):
            latest = c.successors[-1][0]
            self.unanswerable("false_premise", f"Who succeeded {latest} as chief executive of {c.name}?",
                              f"The documents name no successor to {latest}, the latest chief executive of {c.name}.")
        for name in MISSING_PRODUCTS:
            self.unanswerable("nonexistent_entity", f"What is the list price of the {name}?", f"The documents describe no product called {name}.")
        for name, question in MISSING_COMPANIES:
            self.unanswerable("nonexistent_entity", question, f"The documents do not mention a company called {name}.")
        for _, target_name in ACQUISITIONS[:3]:
            acquirer = self.companies[target_name].acquired[0]  # type: ignore[index]
            self.unanswerable("unstated_fact", f"How much did {acquirer} pay for {target_name}?", "The price was not disclosed in the documents.")
        for name in ANNUAL_REPORTERS[:2]:
            self.unanswerable("unstated_fact", f"What was {possessive(name)} revenue in 2025?", f"The documents do not state {possessive(name)} revenue for 2025.")
        for name in ANNUAL_REPORTERS[2:]:
            self.unanswerable("unstated_fact", f"Who chairs the remuneration committee of {name}?", f"The documents do not name a remuneration committee chair for {name}.")
        for decoy_name, _, _ in DECOYS[:2]:
            self.unanswerable("unstated_fact", f"Who is the chief executive of {decoy_name}?", f"The documents do not name a chief executive of {decoy_name}.")
        self.unanswerable("unstated_fact", "How many units did Tessaline Marine ship in 2025?", "The documents contain no shipment figures for Tessaline Marine.")

    # ----- output ----------------------------------------------------------------------------------------------------------------
    def files(self) -> dict[str, str]:
        self.rng.shuffle(self.questions)
        questions, qrels = [], []
        for i, record in enumerate(self.questions, start=1):
            record = {"id": f"h_{i:03d}", **record}
            for doc_id, grade in record.pop("_qrels"):
                qrels.append({"query_id": record["id"], "doc_id": doc_id, "relevance": grade})
            questions.append(record)
        out = {f"docs/{self.ids[key]}.md": doc.text() for key, doc in sorted(self.docs.items(), key=lambda item: self.ids[item[0]])}
        out["questions.jsonl"] = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
        out["qrels.jsonl"] = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qrels)
        return out


def build_suite(seed: int = SEED) -> dict[str, str]:
    """Every file of the suite, keyed by its path under data/hard."""
    suite = Suite(seed)
    suite.build_world()
    suite.write_documents()
    suite.assign_ids()
    suite.write_questions()
    return suite.files()


def stale_files(output: Path, files: dict[str, str]) -> list[str]:
    present = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} if output.is_dir() else set()
    changed = [name for name, content in files.items() if name not in present or (output / name).read_text(encoding="utf-8") != content]
    return sorted(changed + list(present - set(files)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if the output differs from what this script generates")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="directory to write (default: data/hard)")
    args = parser.parse_args(argv)
    files = build_suite()
    stale = stale_files(args.output, files)
    if args.check:
        for name in stale:
            print(f"stale: {name}  (run: python scripts/generate_hard_suite.py)", file=sys.stderr)
        return 1 if stale else 0
    for name in stale:
        path = args.output / name
        if name in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(files[name], encoding="utf-8")
        else:
            path.unlink()
    print(f"{len(files) - 2} documents and {files['questions.jsonl'].count(chr(10))} questions in {args.output} ({len(stale)} file(s) updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
