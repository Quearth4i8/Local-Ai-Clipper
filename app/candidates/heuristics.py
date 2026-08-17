"""Offline (no-LLM) signal detection.

These are *patterns*, not the whole detector. They exist to
  1. rank which parts of a long video the LLM should look at first,
  2. give the LLM structured hints inside the prompt,
  3. keep working when the LLM is unavailable.

Adding a language = adding a key to PATTERNS. Nothing else changes.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..models import Sentence

Rule = tuple  # (compiled_regex, weight)


def _rules(pairs: Sequence[tuple]) -> List[Rule]:
    return [(re.compile(p, re.IGNORECASE | re.UNICODE), w) for p, w in pairs]


# --------------------------------------------------------------------------
# ENGLISH
# --------------------------------------------------------------------------
EN = {
    "hook": _rules([
        (r"\b(nobody|no one) (ever )?(tells?|talks? about|mentions?)\b", 1.0),
        (r"\bthe (biggest|worst|number one|#1|most common) (mistake|problem|myth|lie|reason)\b", 1.0),
        (r"\b(here'?s|this is) (the thing|what (happened|nobody|most people)|why)\b", 0.9),
        (r"\bi (never|didn'?t|couldn'?t|would never)\b", 0.7),
        (r"\b(most|everyone|people) (think|believe|assume)s? (that )?.{0,25}\b(but|however|actually)\b", 0.95),
        (r"\byou'?(re| are) (probably |completely )?(doing|getting) (it|this) wrong\b", 1.0),
        (r"\b(what|why|how|when) (do|does|did|is|are|would|should|can) (you|we|they|people)\b", 0.6),
        (r"\b(let me tell you|listen|imagine|picture this|check this out)\b", 0.6),
        (r"\bthe (truth|reality|secret|trick|problem) is\b", 0.85),
        (r"\b(actually|honestly|literally|genuinely) [a-z]+", 0.3),
        (r"\bit turns out\b", 0.7),
        (r"\b(i'?m going to|i'?ll) (tell|show|explain)\b", 0.5),
        (r"^\s*(so|and) (there'?s|here'?s|the)\b", 0.25),
        (r"\b\d{2,}\s?(percent|%|years?|dollars?|times|people|hours?)\b", 0.5),
        (r"\bnever\b.{0,20}\bagain\b", 0.55),
        (r"\?\s*$", 0.45),
    ]),
    "emotion": _rules([
        (r"\b(amazing|incredible|insane|crazy|unbelievable|shocking|wild|nuts|brutal)\b", 0.8),
        (r"\b(love|hate|scared|terrified|afraid|angry|furious|excited|proud|ashamed|heartbroken)\b", 0.7),
        (r"\b(i (was|felt) (so|really|completely))\b", 0.6),
        (r"\b(devastat|humiliat|traumat|betray)\w*", 0.85),
        (r"\[?\b(laugh(s|ter|ing)?|chuckles?)\b\]?", 0.9),
        (r"\b(ha){2,}\b|\blol\b|\blmao\b", 0.7),
        (r"\b(oh my god|holy|damn|shit|fuck|wtf|jesus)\b", 0.7),
        (r"!{1,}", 0.35),
        (r"\b(worst|best) (day|moment|thing|feeling) (of|in) my life\b", 1.0),
        (r"\b(cried|crying|tears)\b", 0.8),
    ]),
    "value": _rules([
        (r"\bthe (key|trick|secret|point|reason|answer) (is|here is|was)\b", 0.9),
        (r"\b(because|the reason (why|that|is))\b", 0.4),
        (r"\b(so what (happens|you (do|want)) is|what you (need|have) to do)\b", 0.85),
        (r"\b(step (one|two|three|1|2|3)|first(ly)?,|second(ly)?,|finally,)\b", 0.6),
        (r"\b(always|never) (do|use|start|try|forget)\b", 0.6),
        (r"\b(my advice|pro tip|rule of thumb|the framework|the strategy)\b", 0.9),
        (r"\b(studies?|research|data|science) (show|shows|suggests?|found)\b", 0.7),
        (r"\b(here'?s how|this is how|the way (to|you))\b", 0.7),
        (r"\b(learned|realized|discovered|figured out) (that|how|why)\b", 0.65),
    ]),
    "story": _rules([
        (r"\b(one day|back (then|when)|a few (years|months|weeks) ago|at the time)\b", 0.8),
        (r"\b(so i|then i|i remember|when i was)\b", 0.6),
        (r"\b(he|she|they) (said|told me|looked at me|asked me)\b", 0.6),
        (r"\b(suddenly|out of nowhere|all of a sudden|next thing)\b", 0.9),
        (r"\b(and that'?s when|that'?s the moment)\b", 0.95),
        (r"\b(ended up|turned out|it worked|it failed)\b", 0.5),
    ]),
    "curiosity": _rules([
        (r"\b(you'?ll never guess|guess what|wait (for it|until)|but here'?s the (thing|crazy part))\b", 1.0),
        (r"\b(and then|but then)\b.{0,30}\b(happened|realized|found)\b", 0.7),
        (r"\b(nobody knows|no one expected|what (happened|came) next)\b", 0.9),
        (r"\b(i'?ll (get|come) (back|to) (to )?that|more on that)\b", 0.5),
        (r"\b(the (crazy|weird|strange|funny) (thing|part) is)\b", 0.9),
    ]),
    "opinion": _rules([
        (r"\b(i think|i believe|in my opinion|honestly|frankly|to be honest)\b", 0.5),
        (r"\b(that'?s (just )?(wrong|nonsense|bullshit|stupid|ridiculous))\b", 1.0),
        (r"\b(i (completely |totally )?disagree|i'?m not buying)\b", 0.9),
        (r"\b(unpopular opinion|hot take|controversial)\b", 1.0),
        (r"\b(people (are|get) (wrong|confused)|everybody is wrong)\b", 0.85),
        (r"\b(overrated|underrated|a scam|a lie)\b", 0.8),
    ]),
    "payoff": _rules([
        (r"\b(so (the|that'?s|in the end)|in the end|the bottom line|to sum up|which means)\b", 0.8),
        (r"\b(that'?s (why|how|the reason)|and that'?s it)\b", 0.85),
        (r"\b(the result was|it ended up|and it worked|we (doubled|tripled|grew))\b", 0.8),
        (r"\b(lesson|takeaway|moral of the story)\b", 0.95),
    ]),
    "weak_start": _rules([
        (r"^\s*(and|but|so|because|which|that|then|also|plus|anyway|yeah|okay|ok|right|um|uh)\b", 0.8),
        (r"^\s*(he|she|it|they|this|that|those|these)\b", 0.5),
        (r"^\s*(exactly|totally|absolutely|sure)\b\.?\s*$", 1.0),
    ]),
}

# --------------------------------------------------------------------------
# FRENCH
# --------------------------------------------------------------------------
FR = {
    "hook": _rules([
        (r"\b(personne ne (te |vous )?(dit|parle|explique))\b", 1.0),
        (r"\b(la|le) (plus grosse|pire|principale|premi[eè]re) (erreur|probl[eè]me|id[eé]e re[çc]ue)\b", 1.0),
        (r"\b(voil[aà] (le truc|pourquoi)|le truc c'?est que)\b", 0.85),
        (r"\b(je n'?ai jamais|j'?aurais jamais)\b", 0.7),
        (r"\b(la plupart des gens|tout le monde) (pense|croit|s'?imagine)\b.{0,30}\b(mais|en fait|sauf que)\b", 0.95),
        (r"\b(vous (faites|vous y prenez)|tu (fais|t'?y prends)) (tout )?[çc]a mal\b", 1.0),
        (r"\b(pourquoi|comment|qu'?est-ce que|combien) (est-ce que |tu |vous |on |les gens )", 0.6),
        (r"\b(laisse[z]?-moi (te |vous )?(dire|expliquer)|[ée]coute[z]?|imagine[z]?)\b", 0.6),
        (r"\b(la (v[eé]rit[eé]|r[eé]alit[eé]|clé|astuce)|le (secret|probl[eè]me)),? c'?est\b", 0.85),
        (r"\b(en (fait|r[eé]alit[eé])|honn[eê]tement|franchement|s[eé]rieusement)\b", 0.3),
        (r"\b(il s'?av[eè]re|figure[z]?-toi|figurez-vous)\b", 0.7),
        (r"\b\d{2,}\s?(pour ?cent|%|ans?|euros?|fois|personnes|heures?)\b", 0.5),
        (r"\bplus jamais\b", 0.55),
        (r"\?\s*$", 0.45),
    ]),
    "emotion": _rules([
        (r"\b(incroyable|dingue|fou|folle|ouf|hallucinant|choquant|violent|horrible)\b", 0.8),
        (r"\b(j'?ador|je d[eé]test|peur|terrifi|[ée]nerv|furieux|excit[eé]|fier|honte|boulevers)\w*", 0.7),
        (r"\b(j'?[eé]tais (tellement|vraiment|compl[eè]tement))\b", 0.6),
        (r"\[?\b(rires?|rit)\b\]?", 0.9),
        (r"\b(mdr|ptdr|haha)\b", 0.7),
        (r"\b(putain|merde|oh mon dieu|bordel)\b", 0.7),
        (r"!{1,}", 0.35),
        (r"\b(le (pire|meilleur) (jour|moment|truc)) de ma vie\b", 1.0),
        (r"\b(pleur[ée]?|larmes)\b", 0.8),
    ]),
    "value": _rules([
        (r"\b(la (clé|solution|r[eé]ponse)|le (point|truc)),? c'?est\b", 0.9),
        (r"\b(parce que|la raison (pour laquelle|c'?est))\b", 0.4),
        (r"\b(ce qu'?il faut faire|ce que tu dois faire|ce qui se passe c'?est)\b", 0.85),
        (r"\b([eé]tape (une|deux|1|2)|premi[eè]rement|deuxi[eè]mement|enfin,)\b", 0.6),
        (r"\b(toujours|jamais) (faire|utiliser|commencer|oublier)\b", 0.6),
        (r"\b(mon conseil|astuce|la m[eé]thode|la strat[eé]gie)\b", 0.9),
        (r"\b((les )?[eé]tudes?|la recherche|la science) (montre|prouve|dit)\b", 0.7),
        (r"\b(voil[aà] comment|c'?est comme [çc]a qu'?on)\b", 0.7),
        (r"\b(j'?ai (appris|compris|d[eé]couvert|r[eé]alis[eé]))\b", 0.65),
    ]),
    "story": _rules([
        (r"\b(un jour|[aà] l'?[eé]poque|il y a (quelques|deux|trois) (ans|mois|semaines))\b", 0.8),
        (r"\b(donc j'?ai|puis j'?ai|je me souviens|quand j'?[eé]tais)\b", 0.6),
        (r"\b(il|elle|ils) (m'?a dit|m'?ont dit|me regarde|m'?a demand[eé])\b", 0.6),
        (r"\b(d'?un coup|soudain|tout [àa] coup|et l[aà])\b", 0.9),
        (r"\b(c'?est (l[aà]|[aà] ce moment-l[aà]) que)\b", 0.95),
        (r"\b(fini par|[çc]a a march[eé]|[çc]a a foir[eé])\b", 0.5),
    ]),
    "curiosity": _rules([
        (r"\b(tu (vas|ne vas) (jamais )?(deviner|croire)|attends|mais l[aà] le truc c'?est)\b", 1.0),
        (r"\b(et l[aà]|mais l[aà])\b.{0,30}\b(s'?est pass[eé]|j'?ai (compris|trouv[eé]))\b", 0.7),
        (r"\b(personne ne sait|ce qui s'?est pass[eé] apr[eè]s)\b", 0.9),
        (r"\b(j'?y reviens|on en reparle)\b", 0.5),
        (r"\b(le truc (dingue|bizarre|marrant) c'?est)\b", 0.9),
    ]),
    "opinion": _rules([
        (r"\b(je (pense|crois|trouve)|[aà] mon avis|honn[eê]tement|franchement)\b", 0.5),
        (r"\b(c'?est (juste )?(faux|n'?importe quoi|d[eé]bile|ridicule))\b", 1.0),
        (r"\b(je (ne )?suis (pas d'?accord|absolument pas d'?accord))\b", 0.9),
        (r"\b(avis impopulaire|hot take|c'?est cliv|controvers)\w*", 1.0),
        (r"\b(les gens se trompent|tout le monde a tort)\b", 0.85),
        (r"\b(surcot[eé]|sous-cot[eé]|une arnaque|un mensonge)\b", 0.8),
    ]),
    "payoff": _rules([
        (r"\b(donc (au final|en gros)|au final|en r[eé]sum[eé]|ce qui veut dire)\b", 0.8),
        (r"\b(c'?est (pour [çc]a|comme [çc]a) que|et voil[aà])\b", 0.85),
        (r"\b(le r[eé]sultat|au final [çc]a a|on a (doubl[eé]|tripl[eé]))\b", 0.8),
        (r"\b(la le[çc]on|[aà] retenir|la morale)\b", 0.95),
    ]),
    "weak_start": _rules([
        (r"^\s*(et|mais|donc|parce que|qui|que|puis|aussi|enfin|bref|ouais|ok|d'?accord|euh)\b", 0.8),
        (r"^\s*(il|elle|ils|elles|ce|[çc]a|cela|ces)\b", 0.5),
        (r"^\s*(exactement|totalement|carr[eé]ment|ouais)\b\.?\s*$", 1.0),
    ]),
}

PATTERNS: Dict[str, Dict[str, List[Rule]]] = {"en": EN, "fr": FR}
CATEGORIES = ("hook", "emotion", "value", "story", "curiosity", "opinion", "payoff")

# --------------------------------------------------------------------------
# Edge cleanup: lines that must never be the first or last line of a clip.
# --------------------------------------------------------------------------
TRANSITION_OUT = _rules([
    # the conversation moving on - always after the payoff, never part of it
    (r"^\s*(anyway|anyways|so anyway|okay so|ok so|alright|all right|right,? so)\b", 1.0),
    (r"\b(let'?s|let us) (change|move on|talk about|switch|get (in)?to)\b", 1.0),
    (r"\b(we should (probably )?(talk|move|wrap)|moving on|next (question|topic)|"
     r"changing (the )?subject)\b", 1.0),
    (r"\b(coming up|after the break|we'?ll be right back|quick break)\b", 1.0),
    (r"^\s*(bref|enfin bref|bon,? (alors|donc)|du coup on)\b", 1.0),
    (r"\b(on (va|devrait) (parler|passer)|passons|changeons de sujet|"
     r"(petite|courte) pause|on revient)\b", 1.0),
])

FILLER_ONLY = _rules([
    (r"^\s*(yeah|yep|yes|no|nope|right|okay|ok|sure|exactly|totally|absolutely|"
     r"true|wow|hmm|mhm|uh|um|oh)\b[\s,.!?]*$", 1.0),
    (r"^\s*(ouais|oui|non|ok|d'?accord|exactement|carr[eé]ment|tout [aà] fait|"
     r"waouh|hmm|euh|ah|bah)\b[\s,.!?]*$", 1.0),
    (r"^\s*(thanks|thank you|merci)[\s,.!?]*$", 1.0),
])


def is_transition(text: str) -> bool:
    """True if this line is the conversation moving on rather than content."""
    return any(rx.search(text) for rx, _ in TRANSITION_OUT)


def is_filler(text: str) -> bool:
    """True if this line is a bare acknowledgement with no content of its own."""
    return len(text.split()) <= 5 and any(rx.search(text) for rx, _ in FILLER_ONLY)


def rules_for(language: str) -> Dict[str, List[Rule]]:
    lang = (language or "en").lower()[:2]
    base = PATTERNS.get(lang)
    if base is None:
        # Unknown language: English patterns still catch numbers, "?", "!",
        # laughter and loanwords, so they are a usable floor.
        return PATTERNS["en"]
    return base


@dataclass
class SignalProfile:
    """Per-sentence pattern hits, normalised to 0..1."""
    hook: float = 0.0
    emotion: float = 0.0
    value: float = 0.0
    story: float = 0.0
    curiosity: float = 0.0
    opinion: float = 0.0
    payoff: float = 0.0
    weak_start: float = 0.0
    tags: List[str] = field(default_factory=list)

    def total(self) -> float:
        return (self.hook + self.emotion + self.value + self.story
                + self.curiosity + self.opinion + self.payoff)


def _score_rules(text: str, rules: List[Rule]) -> float:
    hits = 0.0
    for rx, weight in rules:
        if rx.search(text):
            hits += weight
    return 1.0 - math.exp(-hits)  # saturating: many weak hits < one strong hit


def profile_sentence(text: str, rules: Dict[str, List[Rule]]) -> SignalProfile:
    p = SignalProfile()
    for cat in CATEGORIES:
        setattr(p, cat, _score_rules(text, rules.get(cat, [])))
    p.weak_start = _score_rules(text, rules.get("weak_start", []))
    p.tags = [c for c in CATEGORIES if getattr(p, c) >= 0.5]
    return p


def profile_all(sentences: List[Sentence], language: str) -> List[SignalProfile]:
    rules = rules_for(language)
    return [profile_sentence(s.text, rules) for s in sentences]


def dominant_type(profiles: Sequence[SignalProfile]) -> str:
    """Best guess at clip type from pattern hits alone (LLM usually overrides)."""
    if not profiles:
        return "other"
    totals = {c: sum(getattr(p, c) for p in profiles) for c in CATEGORIES}
    mapping = {
        "emotion": "emotional", "value": "educational", "story": "story",
        "curiosity": "surprising", "opinion": "opinion", "hook": "surprising",
        "payoff": "educational",
    }
    best = max(totals, key=totals.get)
    return mapping.get(best, "other") if totals[best] > 0.6 else "other"
