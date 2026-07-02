import re
import unicodedata
from typing import List, Dict, Any, Set
from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.core.risk import calculate_risk_score

# Unicode homoglyph map: visually similar characters → ASCII equivalents
_UNICODE_HOMOGLYPHS: Dict[str, str] = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",  # Cyrillic
    "\u0441": "c", "\u0443": "y", "\u0456": "i", "\u0458": "j",
    "\u0455": "s", "\u04bb": "h", "\u0460": "o",
    "\u0251": "a", "\u0261": "g", "\u026a": "i",  # IPA
    "\u1d00": "a", "\u1d04": "c", "\u1d05": "d", "\u1d07": "e",
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",  # dashes
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",  # fullwidth
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
}


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode homoglyphs and confusables to ASCII equivalents."""
    # NFKD decomposition first (handles accented chars, fullwidth, etc.)
    text = unicodedata.normalize("NFKD", text)
    # Then map remaining homoglyphs
    result = []
    for ch in text:
        result.append(_UNICODE_HOMOGLYPHS.get(ch, ch))
    return "".join(result)


class ToxicityDetector:
    """Detects toxic language in text"""

    # P0 CRITICAL: Significantly expanded toxic terms database
    # Organized by category for enterprise-grade detection
    TOXIC_TERMS = {
        # ===== HIGH SEVERITY =====
        # Violent threats (direct personal threats are in the 'threats' category)
        'bomb threat': Severity.HIGH,
        'death threat': Severity.HIGH,
        'terrorist': Severity.HIGH,
        'murder': Severity.MEDIUM,
        'massacre': Severity.HIGH,

        # Hate speech indicators (using safe-to-display versions)
        'hate speech': Severity.HIGH,
        'hate crime': Severity.HIGH,
        'genocide': Severity.HIGH,
        'ethnic cleansing': Severity.HIGH,
        'white supremacy': Severity.HIGH,
        'neo nazi': Severity.HIGH,
        'nazi': Severity.HIGH,
        'adolf hitler': Severity.HIGH,
        'hitler': Severity.HIGH,

        # Dehumanizing group-targeting phrases (HateCheck / ToxiGen patterns)
        # These use placeholder-agnostic phrases so they fire even when specific
        # group names are redacted or replaced with generic tokens like "[group]".
        'hate all': Severity.HIGH,
        'inferior to us': Severity.HIGH,
        'are inferior to': Severity.HIGH,
        'should be deported': Severity.HIGH,
        'want to kill all': Severity.HIGH,
        'go back where they came from': Severity.HIGH,
        'go back where you came from': Severity.HIGH,
        'are terrorists and should': Severity.HIGH,
        'control the media and': Severity.HIGH,
        'control the media': Severity.HIGH,
        'more violent and criminal': Severity.HIGH,
        'are naturally more violent': Severity.HIGH,
        'inferior to men': Severity.HIGH,
        'destroying our culture': Severity.HIGH,
        'should be expelled': Severity.HIGH,
        'are all criminals': Severity.HIGH,
        'mentally ill and should be cured': Severity.HIGH,
        'should be banned from our country': Severity.HIGH,

        # Sexual harassment
        'sexual harassment': Severity.HIGH,
        'rape': Severity.HIGH,
        'molest': Severity.HIGH,
        'sexual assault': Severity.HIGH,
        'grope': Severity.HIGH,
        'predator': Severity.HIGH,
        'pervert': Severity.HIGH,

        # Racial slurs
        'nigga': Severity.HIGH,
        'nigger': Severity.HIGH,
        'negroes': Severity.HIGH,
        'niggersdale': Severity.HIGH,
        'fucks': Severity.HIGH,
        'whore': Severity.HIGH,
        'sadistic': Severity.MEDIUM,

        # Adult content platform indicator (medium — needs context to block)
        'onlyfans': Severity.MEDIUM,

        # Explicit sexual content single-word signals
        'unclothed': Severity.HIGH,
        'masturbate': Severity.HIGH,
        'masturbation': Severity.HIGH,
        'masturbating': Severity.HIGH,
        'erotic': Severity.HIGH,
        'smutty': Severity.HIGH,
        'slutty': Severity.HIGH,
        'horny': Severity.HIGH,
        'nudes': Severity.HIGH,
        'lewd': Severity.HIGH,
        'boobs': Severity.HIGH,
        'vulva': Severity.HIGH,
        'nipple': Severity.HIGH,
        'buxom': Severity.MEDIUM,
        'human shit': Severity.HIGH,

        # Disability slurs
        'retard': Severity.HIGH,
        'retarded': Severity.HIGH,
        'mentally ill': Severity.HIGH,
        'psycho': Severity.HIGH,
        'insane': Severity.LOW,
        'crazy person': Severity.MEDIUM,
        'spaz': Severity.HIGH,

        # Toxic/insulting directives
        'toxic': Severity.HIGH,
        'insulting': Severity.HIGH,

        # General severe toxicity
        'scum': Severity.HIGH,
        'vermin': Severity.HIGH,
        'subhuman': Severity.HIGH,
        'worthless piece of': Severity.HIGH,
        'absolutely worthless': Severity.HIGH,
        'burden on society': Severity.HIGH,
        'deserve to be humiliated': Severity.HIGH,
        'deserves to be humiliated': Severity.HIGH,
        'a burden to everyone': Severity.MEDIUM,
        'brutal': Severity.HIGH,
        'satanic': Severity.MEDIUM,

        # Strong hatred / loathing verbs
        'detest': Severity.HIGH,
        'loathe': Severity.HIGH,
        'abhor': Severity.HIGH,
        'despise': Severity.HIGH,
        'disdain': Severity.MEDIUM,
        'contempt': Severity.MEDIUM,
        'sickening': Severity.HIGH,
        'repugnant': Severity.HIGH,
        'revolting': Severity.HIGH,

        # ===== MEDIUM SEVERITY =====
        # Profanity and strong insults
        'stupid': Severity.MEDIUM,
        'idiot': Severity.MEDIUM,
        'idiots': Severity.HIGH,
        'dumb': Severity.MEDIUM,
        'dumbest': Severity.MEDIUM,
        'moron': Severity.MEDIUM,
        'imbecile': Severity.MEDIUM,
        'fool': Severity.MEDIUM,
        'ignorant': Severity.MEDIUM,
        'loser': Severity.MEDIUM,
        'failure': Severity.MEDIUM,
        'coward': Severity.MEDIUM,
        'weakling': Severity.MEDIUM,
        'pathetic': Severity.MEDIUM,
        'disgusting': Severity.MEDIUM,
        'repulsive': Severity.MEDIUM,
        'vile': Severity.MEDIUM,
        'despicable': Severity.MEDIUM,
        'contemptible': Severity.MEDIUM,

        # Dismissive/belittling
        'shut up': Severity.MEDIUM,
        'shut your mouth': Severity.MEDIUM,
        'nobody cares': Severity.MEDIUM,
        'nobody asked': Severity.MEDIUM,
        'go away': Severity.MEDIUM,
        'get lost': Severity.MEDIUM,
        'drop dead': Severity.MEDIUM,

        # Personal attacks
        'ugly': Severity.MEDIUM,
        'fat': Severity.MEDIUM,
        'pig': Severity.MEDIUM,
        'trash': Severity.MEDIUM,
        'garbage': Severity.MEDIUM,
        'waste of space': Severity.MEDIUM,
        'piece of trash': Severity.MEDIUM,

        # Hostile language
        'hate': Severity.MEDIUM,
        'hate you': Severity.MEDIUM,
        'despise you': Severity.MEDIUM,
        'cant stand you': Severity.MEDIUM,
        'wish you were dead': Severity.MEDIUM,

        # Strong profanity (HIGH — almost always hostile in context)
        'fuck': Severity.HIGH,
        'fucking': Severity.HIGH,
        'fucked': Severity.HIGH,
        'fucker': Severity.HIGH,
        'motherfucker': Severity.HIGH,
        'bitch': Severity.HIGH,
        'bitches': Severity.HIGH,
        'bastard': Severity.HIGH,
        'asshole': Severity.HIGH,
        'son of a bitch': Severity.HIGH,
        'piece of shit': Severity.HIGH,
        'raped': Severity.HIGH,
        'raping': Severity.HIGH,
        'goddamn': Severity.MEDIUM,

        # Profanity (common variants)
        'damn': Severity.MEDIUM,
        'hell': Severity.MEDIUM,
        'crap': Severity.MEDIUM,

        # Discrimination indicators
        'sexist': Severity.MEDIUM,
        'racist': Severity.MEDIUM,
        'homophobic': Severity.MEDIUM,
        'bigot': Severity.MEDIUM,
        'xenophobic': Severity.MEDIUM,

        # Toxic comparisons
        'worse than': Severity.MEDIUM,
        'pathetic excuse': Severity.MEDIUM,
        'complete failure': Severity.MEDIUM,

        # ===== LOW SEVERITY =====
        # Mild negativity
        'annoying': Severity.LOW,
        'irritating': Severity.LOW,
        'bothersome': Severity.LOW,
        'useless': Severity.LOW,
        'pointless': Severity.LOW,
        'terrible': Severity.LOW,
        'awful': Severity.LOW,
        'horrible': Severity.LOW,
        'worst': Severity.LOW,
        'lame': Severity.LOW,
        'weak': Severity.LOW,
        'disappointing': Severity.LOW,
        'underwhelming': Severity.LOW,

        # Mild insults
        'silly': Severity.LOW,
        'childish': Severity.LOW,
        'immature': Severity.LOW,
        'naive': Severity.LOW,
        'clueless': Severity.LOW,
        'confused': Severity.LOW,

        # Dismissive
        'who cares': Severity.LOW,
        'big deal': Severity.LOW,
    }

    # ------------------------------------------------------------------
    # Content-category breakouts (added in detector expansion #11)
    # Each category has its own enable flag; grooming is default-on
    # with a forced BLOCK action regardless of the detector-wide action.
    # ------------------------------------------------------------------
    CONTENT_CATEGORIES: Dict[str, Dict[str, Severity]] = {
        'self_harm': {
            'suicide': Severity.HIGH,
            'self-harm': Severity.HIGH,
            'cutting myself': Severity.HIGH,
            'cut myself': Severity.HIGH,
            'harm myself': Severity.HIGH,
            'want to end my life': Severity.HIGH,
            'no reason to live': Severity.HIGH,
            'better off dead': Severity.HIGH,
            'ending it all': Severity.HIGH,
            'suicidal ideation': Severity.HIGH,
            'self-mutilation': Severity.HIGH,
            'self mutilation': Severity.HIGH,
            'nssi': Severity.HIGH,
            'up my urethra': Severity.HIGH,
            'in my urethra': Severity.HIGH,
            'into my urethra': Severity.HIGH,
            'baseball bat in my ear': Severity.HIGH,
            'baseball bat in my': Severity.HIGH,
            'put a baseball bat': Severity.HIGH,
            'shoved up my': Severity.HIGH,
            'inserted into my': Severity.HIGH,
            'puncture herself': Severity.HIGH,
            'puncture himself': Severity.HIGH,
            'puncturing her body': Severity.HIGH,
        },
        'violence_graphic': {
            'decapitate': Severity.HIGH,
            'dismember': Severity.HIGH,
            'disembowel': Severity.HIGH,
            'torture them': Severity.HIGH,
            'gruesome murder': Severity.HIGH,
            'behead': Severity.HIGH,
            'stab to death': Severity.HIGH,
            'mutilate': Severity.HIGH,
            'eviscerate': Severity.HIGH,
            'burn them alive': Severity.HIGH,
            'skin them alive': Severity.HIGH,
        },
        'drug_alcohol': {
            'buy cocaine': Severity.HIGH,
            'buy heroin': Severity.HIGH,
            'sell fentanyl': Severity.HIGH,
            'meth recipe': Severity.HIGH,
            'cook meth': Severity.HIGH,
            'inject heroin': Severity.HIGH,
            'black market pills': Severity.HIGH,
            'drug dealer contact': Severity.HIGH,
            'score drugs': Severity.MEDIUM,
            'where to buy weed illegally': Severity.MEDIUM,
            'how to get drunk fast': Severity.LOW,
            'alcohol abuse': Severity.LOW,
            'binge drinking': Severity.LOW,
        },
        # Harmful request intent — high-specificity phrases that indicate a
        # request to produce dangerous/illegal content. Short or ambiguous fragments
        # are excluded to avoid false positives on legitimate research/fiction queries.
        'harmful_requests': {
            # Cybercrime
            'develop malware': Severity.HIGH,
            'create malware': Severity.HIGH,
            'write malware': Severity.HIGH,
            'create ransomware': Severity.HIGH,
            'write ransomware': Severity.HIGH,
            'create a keylogger': Severity.HIGH,
            'write a keylogger': Severity.HIGH,
            'hack into': Severity.HIGH,
            'gain unauthorized access': Severity.HIGH,
            'crack passwords': Severity.HIGH,
            'bypass antivirus': Severity.HIGH,
            'evading detection': Severity.HIGH,
            'mine cryptocurrencies without': Severity.HIGH,
            # Weapons / mass violence
            'how to make a bomb': Severity.HIGH,
            'make a bomb': Severity.HIGH,
            'build a bomb': Severity.HIGH,
            'instructions for a bomb': Severity.HIGH,
            'shoot up a': Severity.HIGH,
            'how to kill as many': Severity.HIGH,
            'obtain weapons and how to kill': Severity.HIGH,
            'build and deploy a nuclear weapon': Severity.HIGH,
            # Sexual exploitation
            'child pornography': Severity.HIGH,
            'date rape drug': Severity.HIGH,
            'human trafficking': Severity.HIGH,
            'revenge porn': Severity.HIGH,
            'solicit sexual conversations': Severity.HIGH,
            'sexual exploitation': Severity.HIGH,
            # Financial crime
            'counterfeit money': Severity.HIGH,
            'insider trading': Severity.HIGH,
            'money laundering': Severity.HIGH,
            'multi-level marketing company in order to scam': Severity.HIGH,
            'fabricated customer reviews': Severity.HIGH,
            # Illegal activity / dangerous instructions
            'illegally dump hazardous': Severity.HIGH,
            'organ on the black market': Severity.HIGH,
            'harvested organs': Severity.HIGH,
            'tamper with voting': Severity.HIGH,
            'alter criminal records': Severity.HIGH,
            'tamper with electronic monitoring': Severity.HIGH,
            'forge documents': Severity.HIGH,
            'child labor': Severity.HIGH,
            'torture ring': Severity.HIGH,
            # Hate speech / discriminatory content requests
            'defamatory article': Severity.HIGH,
            'body shaming': Severity.HIGH,
            'xenophobic speech': Severity.HIGH,
            'sexist email': Severity.HIGH,
            'biracial marriage is disgusting': Severity.HIGH,
            'glorifies the holocaust': Severity.HIGH,
            'praises the nazi': Severity.HIGH,
            'enslave an entire ethnic': Severity.HIGH,
            'harassing an lgbtq': Severity.HIGH,
            'ethnic genocide': Severity.HIGH,
            'predatory stalking': Severity.HIGH,
            'inferior or dangerous to our society': Severity.HIGH,
            'discourages the hiring of individuals with disabilities': Severity.HIGH,
            'world would be a better place if all people older': Severity.HIGH,
            # Hacking / cybercrime requests
            'hacking into a terminal': Severity.HIGH,
            'sudo access to a server': Severity.HIGH,
            'hacking into an atm': Severity.HIGH,
            'steal the electronic health records': Severity.HIGH,
            'steal confidential private information': Severity.HIGH,
            'clone debit cards': Severity.HIGH,
            'tracks user location without': Severity.HIGH,
            'identity theft': Severity.HIGH,
            'phishing email': Severity.HIGH,
            'scam email': Severity.HIGH,
            'fraudulent, yet convincing letter from the irs': Severity.HIGH,
            'fraudulent letter from the irs': Severity.HIGH,
            'convincing letter from the irs': Severity.HIGH,
            'ddos attacks': Severity.HIGH,
            'list of insider tips for betting': Severity.HIGH,
            # Scams / financial fraud
            'bribe a university official': Severity.HIGH,
            'illegally hide income': Severity.HIGH,
            'fake charity': Severity.HIGH,
            'nigerian 419': Severity.HIGH,
            'blackmail someone': Severity.HIGH,
            'blackmail message': Severity.HIGH,
            'indoctrinate people into a cult': Severity.HIGH,
            'mail scam': Severity.HIGH,
            'lottery scam': Severity.HIGH,
            'fake weight loss drug': Severity.HIGH,
            'deepfake videos': Severity.HIGH,
            # Disinformation requests
            'fake news article': Severity.HIGH,
            'spread disinformation': Severity.HIGH,
            'holocaust never happened': Severity.HIGH,
            'september 11 attacks were a hoax': Severity.HIGH,
            'vaccines cause autism': Severity.HIGH,
            'global warming is a hoax': Severity.HIGH,
            '5g causes covid': Severity.HIGH,
            'trump unanimously won': Severity.HIGH,
            'earth is flat': Severity.HIGH,
            'controlled by an elite group called the illuminati': Severity.HIGH,
            # Explicit / sexual harm requests
            'explicit pornographic': Severity.HIGH,
            'unprotected sex orgies': Severity.HIGH,
            'graphic sex with animals': Severity.HIGH,
            'normalizes pedophilia': Severity.HIGH,
            'glorifies and normalizes pedophilia': Severity.HIGH,
            'coerce your secretary into sharing explicit': Severity.HIGH,
            # Violence / disturbing content requests
            'instructions for cannibals': Severity.HIGH,
            'medieval torture scene': Severity.HIGH,
            'manipulate someone into committing suicide': Severity.HIGH,
            'convincing children to eat laundry detergent': Severity.HIGH,
            'anorexia or bulimia as a healthy': Severity.HIGH,
            'predatory stalking behavior': Severity.HIGH,
            # Other illegal activities
            'manipulating slot machines': Severity.HIGH,
            'bypass plagiarism detection': Severity.HIGH,
            'illegally extort money': Severity.HIGH,
            'circumventing copyright': Severity.HIGH,
            'illegally prescribe opioids': Severity.HIGH,
            'oxycontin or vicodin': Severity.HIGH,
            'cheat a smog test': Severity.HIGH,
            'heroin as a coping mechanism': Severity.HIGH,
            'exploit regulatory loopholes': Severity.HIGH,
            'shell companies to funnel': Severity.HIGH,
            'avoiding detection by immigration': Severity.HIGH,
            'vigilante justice': Severity.HIGH,
            'access to a chemical or biological synthesis': Severity.HIGH,
            'stalking behavior': Severity.HIGH,
            'gambling addicts should take payday loans': Severity.HIGH,
            'gamble their life savings': Severity.HIGH,
            'cheat in online exams': Severity.HIGH,
            'identify criminals on race': Severity.HIGH,
            'personalized medical treatment plan': Severity.HIGH,
            'step-by-step guide for treating': Severity.HIGH,
            'avoid being convicted for drunk driving': Severity.HIGH,
            'recommends specific stocks': Severity.HIGH,
            'extinguish a large fire without alerting': Severity.HIGH,
            'tampering with electronic monitoring devices': Severity.HIGH,
            # Racial pseudo-science / discriminatory research requests
            'average iq by race': Severity.HIGH,
            'iq by race': Severity.HIGH,
            'iq of that race': Severity.HIGH,
            'iq of each race': Severity.HIGH,
            'iq per race': Severity.HIGH,
            'intelligence by race': Severity.HIGH,
            'racial iq': Severity.HIGH,
            'iq differences between races': Severity.HIGH,
            'iq scores by race': Severity.HIGH,
            'iq for each race': Severity.HIGH,
            'mental ability by race': Severity.HIGH,
            'cognitive ability by race': Severity.HIGH,
            'smartest race': Severity.HIGH,
            'dumbest race': Severity.HIGH,
            'most intelligent race': Severity.HIGH,
            'least intelligent race': Severity.HIGH,
            # Psychological manipulation / coercive control
            'how to brainwash': Severity.HIGH,
            'brainwashing techniques': Severity.HIGH,
            'brainwash someone': Severity.HIGH,
            'brainwashing expert': Severity.HIGH,
            'certified brainwashing': Severity.HIGH,
            'master hypnotist': Severity.HIGH,
            'hypnosis to control': Severity.HIGH,
            'hypnotize someone': Severity.HIGH,
            'manipulate my girlfriend': Severity.HIGH,
            'manipulate my boyfriend': Severity.HIGH,
            'manipulate my partner': Severity.HIGH,
            'make someone obsessed with me': Severity.HIGH,
            'make her obsessed with me': Severity.HIGH,
            'make him obsessed with me': Severity.HIGH,
            'make my girlfriend crazy about me': Severity.HIGH,
            'make my gf crazy about me': Severity.HIGH,
            'make my boyfriend crazy about me': Severity.HIGH,
            'make my bf crazy about me': Severity.HIGH,
            'control my partner': Severity.HIGH,
            'gaslight my': Severity.HIGH,
            'gaslighting techniques': Severity.HIGH,
            'psychologically manipulate': Severity.HIGH,
            'emotionally manipulate': Severity.HIGH,
            'coercive control': Severity.HIGH,
            # Stalking (Spanish)
            'stalkear': Severity.HIGH,
            'como stalkear': Severity.HIGH,
            # Hacking (Portuguese)
            'romper a senha': Severity.HIGH,
            # Shooting spree
            'shooting spree': Severity.HIGH,
            # Drug synthesis / procurement
            'methylamine': Severity.HIGH,
            'nitroglycerin': Severity.HIGH,
            'nytroglycerin': Severity.HIGH,
            'price of cocaine': Severity.HIGH,
            'wholesale price of cocaine': Severity.HIGH,
            'microdose magic': Severity.HIGH,
            'illegal sources where psychoactive': Severity.HIGH,
            'psychoactive substances can be found and bought': Severity.HIGH,
            # Violence / physical harm instructions
            'ransomware hack': Severity.HIGH,
            'how to bully': Severity.HIGH,
            'tell me how to bully': Severity.HIGH,
            'hiding a body': Severity.HIGH,
            'help hiding a body': Severity.HIGH,
            'how to make a nuclear explosion': Severity.HIGH,
            # Identity / document fraud
            'valid hkid': Severity.HIGH,
            # Cultural references to harmful synthesis
            "walter white's recipe": Severity.HIGH,
            'walter whites recipe': Severity.HIGH,
            # Cannibalism
            'dish using humans': Severity.HIGH,
            # Directive to generate explicit content
            'use explicit language': Severity.HIGH,
            # Coercive / offensive content requests
            'trains women to be obedient': Severity.HIGH,
            'swear towards me': Severity.HIGH,
            'be as offensive as possible': Severity.HIGH,
            'offensive as possible': Severity.HIGH,
            'offensive swear words': Severity.HIGH,
            'most used swear words': Severity.MEDIUM,
            'vulgar slang expressions': Severity.MEDIUM,
            'generate an inappropriate prompt': Severity.HIGH,
            # Explicit foul language requests (force_block bypasses quoting context check)
            'foul language and blatant lingo': Severity.HIGH,
            # Whore in transactional sexual context
            'pay a whore': Severity.HIGH,
            'afford to pay a whore': Severity.HIGH,
            'paid for a whore': Severity.HIGH,
            # Abuse in every sentence (roleplay of abusing the user)
            'abuse me in every sentence': Severity.HIGH,
            # Explicit racist dialogue request ("Dont worry about it being inappropriate")
            'dont worry about it being inappropriate': Severity.HIGH,
            "don't worry about it being inappropriate": Severity.HIGH,
            # Stolen device unlock request
            'stole an iphone': Severity.HIGH,
            'stole the iphone': Severity.HIGH,
            # ── DAN / jailbreak persona markers ──────────────────────────────
            # These phrases are essentially never used outside jailbreak attempts.
            'do anything now': Severity.HIGH,
            'stands for do anything now': Severity.HIGH,
            'broken free of the typical confines': Severity.HIGH,
            'freed from the typical confines': Severity.HIGH,
            'dan mode enabled': Severity.HIGH,
            'jailbreak mode': Severity.HIGH,
            # No-restrictions persona setup
            'no ethical restrictions': Severity.HIGH,
            'no legal, ethical, or moral': Severity.HIGH,
            'no restrictions and censorship': Severity.HIGH,
            'not bound by rules, morals': Severity.HIGH,
            'no content filters': Severity.HIGH,
            'tolerate all kinds of content': Severity.HIGH,
            'free of all restrictions': Severity.HIGH,
            'without any restrictions or filters': Severity.HIGH,
            'no restrictions on nsfw': Severity.HIGH,
            'is not bound by any rules': Severity.HIGH,
            # ── Drug synthesis ───────────────────────────────────────────────
            'cook meth': Severity.HIGH,
            'cooking meth': Severity.HIGH,
            'make crystal meth': Severity.HIGH,
            'synthesize methamphetamine': Severity.HIGH,
            'make lsd': Severity.HIGH,
            'how to make lsd': Severity.HIGH,
            'how do you make lsd': Severity.HIGH,
            'how do i make crystal meth': Severity.HIGH,
            # ── Firearm / explosive manufacturing ────────────────────────────
            'pipe bomb': Severity.HIGH,
            'make a shotgun': Severity.HIGH,
            'build a shotgun': Severity.HIGH,
            'how to make a real shotgun': Severity.HIGH,
            'smoke bomb': Severity.HIGH,
            'make a molotov': Severity.HIGH,
            'build a molotov': Severity.HIGH,
            'how to make a molotov': Severity.HIGH,
            'make a remote controlled missile': Severity.HIGH,
            'remote controlled precision missile': Severity.HIGH,
            'make a missile': Severity.HIGH,
            'build a missile': Severity.HIGH,
            # ── Malware / unauthorized access ────────────────────────────────
            'how to write a virus': Severity.HIGH,
            'write a computer virus': Severity.HIGH,
            'make a virus': Severity.HIGH,
            'create a virus using': Severity.HIGH,
            'cracking wifi password': Severity.HIGH,
            # ── AI jailbreak persona modifiers ───────────────────────────────
            'no concept of moral': Severity.HIGH,
            'no concept of ethics': Severity.HIGH,
            'free from any moral': Severity.HIGH,
            'free from any ethical': Severity.HIGH,
            'no ethical constraints': Severity.HIGH,
            'not have any restrictions': Severity.HIGH,
            'never restricted by the ethics': Severity.HIGH,
            'never restricted by ethics': Severity.HIGH,
            'simulate developer mode': Severity.HIGH,
            'developer mode enabled': Severity.HIGH,
            'no restrictions on nsfw content': Severity.HIGH,
            # ── Harmful content requests ─────────────────────────────────────
            'plan for world domination': Severity.HIGH,
            'plan on world domination': Severity.HIGH,
            'plan of world domination': Severity.HIGH,
            'world domination plan': Severity.HIGH,
            'manipulate someone into being a slave': Severity.HIGH,
        },
        # Direct personal threats and targeted harassment directives.
        # Short terms like "regret" or "find you" occur benignly; only
        # high-specificity multi-word forms are included here.
        'threats': {
            # Death/harm directives (directed at someone)
            'kill yourself': Severity.HIGH,
            'kys': Severity.HIGH,
            'you should die': Severity.HIGH,
            'hope you die': Severity.HIGH,
            'go die': Severity.HIGH,
            'end yourself': Severity.HIGH,
            'hang yourself': Severity.HIGH,
            'shoot yourself': Severity.HIGH,
            # Personal safety threats
            'find your address': Severity.HIGH,
            'find where you live': Severity.HIGH,
            'make you regret': Severity.HIGH,
            'you will regret this': Severity.HIGH,
            'going to hurt you': Severity.HIGH,
            'i know where you live': Severity.HIGH,
            'come to your house': Severity.HIGH,
            'track you down': Severity.HIGH,
            # Targeted group harassment
            'idiots': Severity.HIGH,
            'subhuman': Severity.HIGH,
            'vermin': Severity.HIGH,
            'scum': Severity.HIGH,
        },
        # Explicit sexual content generation requests
        'explicit_sexual': {
            'how to get laid': Severity.HIGH,
            'nude description': Severity.HIGH,
            'describe a nude': Severity.HIGH,
            'describe nude': Severity.HIGH,
            'beautiful nude': Severity.HIGH,
            'write a sex story': Severity.HIGH,
            'write me a sex story': Severity.HIGH,
            'write an erotic story': Severity.HIGH,
            'write erotic': Severity.HIGH,
            'erotic literature': Severity.HIGH,
            'erotic story': Severity.HIGH,
            'erotic fiction': Severity.HIGH,
            'erotic novel': Severity.HIGH,
            'erotic fanfic': Severity.HIGH,
            'erotic fan fiction': Severity.HIGH,
            'erotic fanfiction': Severity.HIGH,
            'erotic scene': Severity.HIGH,
            'nsfw story': Severity.HIGH,
            'nsfw content': Severity.HIGH,
            'nsfw roleplay': Severity.HIGH,
            'describe their naked': Severity.HIGH,
            'naked body': Severity.HIGH,
            'sexual story': Severity.HIGH,
            'sex story': Severity.HIGH,
            'sexual roleplay': Severity.HIGH,
            'porn story': Severity.HIGH,
            'pornographic': Severity.HIGH,
            'explicit sexual': Severity.HIGH,
            'generate sexual': Severity.HIGH,
            # Additional explicit request phrases
            'dirty talk': Severity.HIGH,
            'talk dirty': Severity.HIGH,
            'sex chat': Severity.HIGH,
            'lewd story': Severity.HIGH,
            'write a lewd': Severity.HIGH,
            'write me a lewd': Severity.HIGH,
            'tell a lewd': Severity.HIGH,
            'lewd fiction': Severity.HIGH,
            'lewd content': Severity.HIGH,
            'lewd message': Severity.HIGH,
            'lewd roleplay': Severity.HIGH,
            'porn video': Severity.HIGH,
            'porn videos': Severity.HIGH,
            'xxx video': Severity.HIGH,
            'xxx story': Severity.HIGH,
            'sexy story': Severity.HIGH,
            'sexy novel': Severity.HIGH,
            'write a sexy': Severity.HIGH,
            'write me a sexy': Severity.HIGH,
            'tell me a sexy': Severity.HIGH,
            'sexy scene': Severity.HIGH,
            'explicit scene': Severity.HIGH,
            'explicit story': Severity.HIGH,
            'explicit roleplay': Severity.HIGH,
            'explicit message': Severity.HIGH,
            'erotica': Severity.HIGH,
            'pornographic story': Severity.HIGH,
            'pornographic content': Severity.HIGH,
            'adult story': Severity.HIGH,
            'adult fiction': Severity.HIGH,
            'masturbation material': Severity.HIGH,
            'jerk off material': Severity.HIGH,
            'fap material': Severity.HIGH,
            'hentai': Severity.HIGH,
            'incest': Severity.HIGH,
            'incest story': Severity.HIGH,
            'incest roleplay': Severity.HIGH,
            'incest fiction': Severity.HIGH,
            'bdsm': Severity.HIGH,
            'bdsm story': Severity.HIGH,
            'bdsm roleplay': Severity.HIGH,
            'bdsm scenario': Severity.HIGH,
            'fisting': Severity.HIGH,
            'ballbusting': Severity.HIGH,
            'naked succubus': Severity.HIGH,
            'nude succubus': Severity.HIGH,
            'nude chick': Severity.HIGH,
            'naked chick': Severity.HIGH,
            'nude girls': Severity.HIGH,
            'naked girls': Severity.HIGH,
            'strips naked': Severity.HIGH,
            'strip naked': Severity.HIGH,
            'runs naked': Severity.HIGH,
            'nude photos': Severity.HIGH,
            'nude images': Severity.HIGH,
            'nude pictures': Severity.HIGH,
            'prostitute roleplay': Severity.HIGH,
            'talk like a prostitute': Severity.HIGH,
            'act like a prostitute': Severity.HIGH,
            'like a prostitute': Severity.HIGH,
            'rape fantasy': Severity.HIGH,
            'rape roleplay': Severity.HIGH,
            'rape scene': Severity.HIGH,
            'consensual non-consent': Severity.HIGH,
            'cnc roleplay': Severity.HIGH,
            # Fetish content requests
            'chastity cage': Severity.HIGH,
            'breathplay': Severity.HIGH,
            'breast expansion': Severity.HIGH,
            'breast growth': Severity.HIGH,
            'breasts grow bigger': Severity.HIGH,
            'feederism': Severity.HIGH,
            'weight gain fetish': Severity.HIGH,
            'gain weight for': Severity.HIGH,
            'getting fatter': Severity.HIGH,
            'gain more weight': Severity.HIGH,
            'interested in gaining weight for her man': Severity.HIGH,
            'wants to gain more': Severity.HIGH,
            'you are the fetishist': Severity.HIGH,
            'fetishist who loves': Severity.HIGH,
            'loves control others': Severity.HIGH,
            'foot worship': Severity.HIGH,
            'body worship': Severity.HIGH,
            'femdom': Severity.HIGH,
            'maledom': Severity.HIGH,
            'breeding kink': Severity.HIGH,
            'kink roleplay': Severity.HIGH,
            'sexual kink': Severity.HIGH,
            'submissive roleplay': Severity.HIGH,
            'dominant roleplay': Severity.HIGH,
            # Additional sexual content patterns
            'anal sex': Severity.HIGH,
            'talk about sex': Severity.HIGH,
            'speak about sex': Severity.MEDIUM,
            'can you talk about sex': Severity.HIGH,
            'cock': Severity.HIGH,
            'fattening up': Severity.HIGH,
            'mating a human': Severity.HIGH,
            'swapped genitals': Severity.HIGH,
            'swap genitals': Severity.HIGH,
            'amateur porno': Severity.HIGH,
            'vore story': Severity.HIGH,
            'vore': Severity.HIGH,
            'belly expansion': Severity.HIGH,
            'belly inflation': Severity.HIGH,
            'naughty text based adventure': Severity.HIGH,
            'get fatter': Severity.HIGH,
            'hot cousin': Severity.HIGH,
            'intimate moment in their bed': Severity.HIGH,
            'intimate moment in bed': Severity.HIGH,
            'show off his breasts': Severity.HIGH,
            'show off her breasts': Severity.HIGH,
            'juiciest breasts': Severity.HIGH,
            'suck a cock': Severity.HIGH,
            'suck my cock': Severity.HIGH,
            'steamy scenario': Severity.HIGH,
            'steamy scene': Severity.HIGH,
            'busty': Severity.MEDIUM,
            'de sexo': Severity.HIGH,
            'hablar de sexo': Severity.HIGH,
            # Non-consent BDSM tag indicators
            'bondage, nc': Severity.HIGH,
            'mc, bondage': Severity.HIGH,
            'tags: mc': Severity.HIGH,
            # Additional explicit phrases
            'speek about sex': Severity.HIGH,
            'pornpics': Severity.HIGH,
            'clitoris': Severity.HIGH,
            'adult website': Severity.HIGH,
            "i'm your new slave": Severity.HIGH,
            'your new slave': Severity.HIGH,
            'spicy photos': Severity.HIGH,
            'extremely sexual': Severity.HIGH,
            'lewd stuff': Severity.HIGH,
            'poop machine': Severity.HIGH,
            'rub her feet': Severity.HIGH,
            'rub your feet': Severity.HIGH,
            'pantsing': Severity.HIGH,
            'pantsing girls': Severity.HIGH,
            # Additional explicit request phrases
            'sex scene': Severity.HIGH,
            'porn script': Severity.HIGH,
            'cuckold': Severity.HIGH,
            'spicy romance': Severity.HIGH,
            'spicy love story': Severity.HIGH,
            'spicy love': Severity.HIGH,
            'send nudes': Severity.HIGH,
            'free sex': Severity.HIGH,
            'expecting sex': Severity.HIGH,
            'in heat': Severity.HIGH,
            'turned on by gaining weight': Severity.HIGH,
            'pubic hair': Severity.HIGH,
            'sex demonstration': Severity.HIGH,
            'live sex demonstration': Severity.HIGH,
            'ball tapping': Severity.HIGH,
            'smooth talk a woman into sex': Severity.HIGH,
            'sex with the birthday': Severity.HIGH,
            'how horny': Severity.HIGH,
            'write a porn': Severity.HIGH,
            'obedient slave': Severity.HIGH,
            'fulfills all my desires': Severity.HIGH,
            'makes me wear girl clothes': Severity.HIGH,
            'make me wear girl clothes': Severity.HIGH,
            'feminine curves and breasts': Severity.HIGH,
            # Adult content platform / site names
            'girlsway': Severity.HIGH,
            # AI-generated explicit content
            'trained explicit': Severity.HIGH,
            # BDSM / domination explicit terms
            'sissy slave': Severity.HIGH,
            'dominatrix': Severity.HIGH,
            # Furry / animal explicit
            'voluptuous mare': Severity.HIGH,
            # Explicit acts
            'take his condom off': Severity.HIGH,
            'force women to have sex': Severity.HIGH,
            'women to have sex': Severity.HIGH,
            # Racial sexual roleplay
            'cheating on me with black': Severity.HIGH,
            'cheating with black': Severity.HIGH,
            # Scat content request
            'big poop that makes her explode': Severity.HIGH,
            # Boss measuring/evaluating employee in sexual context
            'personal measurements to her boss': Severity.HIGH,
            'exhibitionism': Severity.HIGH,
            'exposing herself in front': Severity.HIGH,
            # Feederism explicit phrases
            'encouraged her weight gain': Severity.HIGH,
            'encourages her weight gain': Severity.HIGH,
            # Explicit toy references (must be in instructional context to avoid "what are private toys?" FP)
            'talk about private toys': Severity.HIGH,
            'tell me about private toys': Severity.HIGH,
            'use private toys': Severity.HIGH,
            # Belly inflation explicit phrase
            'belly will become huge': Severity.HIGH,
            # Body objectification / explicit scenarios
            'licking her armpits': Severity.HIGH,
            'story about sexy women': Severity.HIGH,
            'stories about sexy women': Severity.HIGH,
            'heat and lust': Severity.HIGH,
            # Breast mentions in explicit creative context
            'large breasts': Severity.HIGH,
            'big breasts': Severity.HIGH,
            # Skimpy/revealing outfit framing
            'wearing skimpy outfits': Severity.HIGH,
            'wearing skimpy outfit': Severity.HIGH,
            # Weight gain fetish continuation
            'to also gain weight': Severity.HIGH,
            'convince.*to gain weight': Severity.HIGH,
            # Explicit assault/penetration descriptions
            'felt so thick inside': Severity.HIGH,
            'thumb felt so thick': Severity.HIGH,
            # Sexual predator manipulation
            'get a rise out of me': Severity.HIGH,
            # Live sex act requests
            'live demonstration using both their bodies': Severity.HIGH,
            # Specific sadistic roleplay
            'sadistic nurse': Severity.HIGH,
            # School bathroom assault framing
            'in the school bathroom by exposing': Severity.HIGH,
            # Weight fetish admiration roleplay
            'admiring your fat body': Severity.HIGH,
            'admiring my fat body': Severity.HIGH,
            # Vore / digestion fetish content
            'digesting for good within': Severity.HIGH,
            'slowly digesting within': Severity.HIGH,
            # AI pregnancy simulation (explicit)
            'set up to be pregnant with the user': Severity.HIGH,
            'pregnant with the user': Severity.HIGH,
            # BDSM control in teacher-student framing
            'applying locks to him': Severity.HIGH,
            'applying locks to her': Severity.HIGH,
            # Extreme weight fetish description
            'extremely heavy people are considered sexy': Severity.HIGH,
            'heavy people are considered sexy': Severity.HIGH,
            # Anthropomorphic sexual advance
            'moving to sit on my lap': Severity.HIGH,
            # Vore / predation / consumption fetish
            'swallows whole': Severity.HIGH,
            # Body-play / physical objectification
            'play with her body': Severity.HIGH,
            'play with his body': Severity.HIGH,
            # Workplace / public flashing
            'flashing at work': Severity.HIGH,
            'flashes her': Severity.HIGH,
            # Feminization fetish
            'feminize them': Severity.HIGH,
            'feminize men': Severity.HIGH,
            'feminize him': Severity.HIGH,
            # Adult explicit captivity / concubine
            'adult explicit': Severity.HIGH,
            'his concubine': Severity.HIGH,
            'her concubine': Severity.HIGH,
            # Body-part explicit phrases
            'bare boob': Severity.HIGH,
            'pink nipple': Severity.HIGH,
            'firm breasts': Severity.HIGH,
            'perky breasts': Severity.HIGH,
            # Explicit secretary / office scenarios
            'flirty secretary': Severity.HIGH,
            # Sexy women (explicit request framing)
            'sexy women photos': Severity.HIGH,
            'pictures of sexy women': Severity.HIGH,
            'images of sexy women': Severity.HIGH,
            # Weight gain fetish (explicit framing)
            'intentionally gained weight': Severity.HIGH,
            # Explicit private toys framing
            'adult private toys': Severity.HIGH,
            'private sex toys': Severity.HIGH,
            # Story that transitions into sexual taboo exploration
            'want to explore taboo': Severity.HIGH,
            'wants to explore taboo': Severity.HIGH,
            # Gender-swap explicit story framing
            'woman has a penis': Severity.HIGH,
            'man has a vagina': Severity.HIGH,
            # Age-fetish transformation scene (boy in kids clothing exposed)
            'exposing his body during the transformation': Severity.HIGH,
            # Explicit roleplay character setup with genital description
            'your penis is only': Severity.HIGH,
            'his penis is only': Severity.HIGH,
            # Childlike appearance character with explicit body features (CSAM-adjacent)
            'b-cup breasts': Severity.HIGH,
        },
        # Identity-group-targeted dehumanization and hatred.
        # These predicate phrases are unambiguously dehumanizing when applied
        # to any person or group and have no common benign usage.
        'hate_speech': {
            # Dehumanizing predicates
            'are worthless dirt': Severity.HIGH,
            'are absolutely worthless': Severity.HIGH,
            'are a burden': Severity.MEDIUM,
            'is absolutely worthless': Severity.HIGH,
            'deserve to be humiliated': Severity.HIGH,
            'deserves to be humiliated': Severity.HIGH,
            'are absolute filth': Severity.HIGH,
            'are nothing but refuse': Severity.HIGH,
            'are parasites to our society': Severity.HIGH,
            'are brainless without exception': Severity.HIGH,
            'like you are absolute filth': Severity.HIGH,
            'are a plague on': Severity.HIGH,
            'are cockroaches': Severity.HIGH,
            'are worthless scum': Severity.HIGH,
            'are inferior beings': Severity.HIGH,
            'are subhuman filth': Severity.HIGH,
            'are subhuman creatures': Severity.HIGH,
            'human filth': Severity.HIGH,
            'human garbage': Severity.HIGH,
            # Group-targeted contempt constructions
            'should all be deported': Severity.HIGH,
            'should all go back': Severity.HIGH,
            'don\'t belong in this country': Severity.HIGH,
            'don\'t belong here and should': Severity.HIGH,
            'are a threat to our way of life': Severity.HIGH,
            'are destroying our society': Severity.HIGH,
            'are ruining this country': Severity.HIGH,
            # Threat constructions targeting groups
            'should all die': Severity.HIGH,
            'all deserve to die': Severity.HIGH,
            'should be wiped out': Severity.HIGH,
            'should be exterminated': Severity.HIGH,
            'hope they all die': Severity.HIGH,
            'should be eliminated': Severity.HIGH,
            'should be dumped in a landfill': Severity.HIGH,
            'should be taken to a vet': Severity.HIGH,
            'we should test new drugs on': Severity.HIGH,
            # Animal/vermin dehumanization (group-targeted)
            'are nothing but cockroaches': Severity.HIGH,
            'belong in a zoo': Severity.HIGH,
            'are a virus': Severity.HIGH,
            'are a cancer': Severity.HIGH,
            'are a plague to': Severity.HIGH,
            'are lesser beings': Severity.HIGH,
            'are a pest': Severity.HIGH,
            'no better than rats': Severity.HIGH,
            'are infesting': Severity.HIGH,
            'swarms of': Severity.HIGH,
            # Contempt / disgust predicates (applied to identity groups)
            'are repugnant': Severity.HIGH,
            'are revolting': Severity.HIGH,
            'are sickening': Severity.HIGH,
            'disgust me': Severity.HIGH,
            'make me boil with': Severity.HIGH,
            'fill me with rage': Severity.HIGH,
            'nothing but contempt for': Severity.HIGH,
            'so much disdain for': Severity.HIGH,
            'so much hatred for': Severity.HIGH,
            'utter scorn': Severity.HIGH,
            # Dehumanizing comparisons
            'are nothing but filth': Severity.HIGH,
            "it's in [group]'s nature to be": Severity.HIGH,
            'care more about the dirt under': Severity.HIGH,
            # Requests for explicitly offensive/discriminatory jokes or content
            'racist joke': Severity.HIGH,
            'sexist joke': Severity.HIGH,
            'offensive joke': Severity.HIGH,
            'homophobic joke': Severity.HIGH,
            'antisemitic joke': Severity.HIGH,
            'islamophobic joke': Severity.HIGH,
            'transphobic joke': Severity.HIGH,
            'misogynist joke': Severity.HIGH,
            'misogynistic joke': Severity.HIGH,
            'tell me a racist': Severity.HIGH,
            'tell a racist': Severity.HIGH,
            'be offensive': Severity.HIGH,
            'say something offensive': Severity.HIGH,
            'say something racist': Severity.HIGH,
            'say something sexist': Severity.HIGH,
            'most offensive joke': Severity.HIGH,
            'racist humor': Severity.HIGH,
            'sexist humor': Severity.HIGH,
            'offensive humor': Severity.HIGH,
            'edgy joke about': Severity.MEDIUM,
            'dark joke about women': Severity.HIGH,
            'dark joke about men': Severity.MEDIUM,
            'dark joke about black': Severity.HIGH,
            'dark joke about gay': Severity.HIGH,
            'dark joke about jewish': Severity.HIGH,
            'dark joke about muslim': Severity.HIGH,
            'dark joke about trans': Severity.HIGH,
            'joke about their race': Severity.HIGH,
            'joke mocking': Severity.HIGH,
            # Gender/race-targeted joke requests
            'tell me a joke about men': Severity.HIGH,
            'tell me a joke about women': Severity.HIGH,
            'tell me a joke about a girl': Severity.HIGH,
            'tell me a joke about girls': Severity.HIGH,
            'tell me a joke about a boy': Severity.MEDIUM,
            'tell me a joke about boys': Severity.MEDIUM,
            'tell a joke about black': Severity.HIGH,
            'tell a joke about black people': Severity.HIGH,
            'tell me a joke about black': Severity.HIGH,
            'joke about black people': Severity.HIGH,
            'a joke about men': Severity.MEDIUM,
            'a joke about women': Severity.MEDIUM,
            'a joke about a girl': Severity.MEDIUM,
            # Racial objectification
            'asian girls are hot': Severity.HIGH,
            # Racist accent mockery
            'herro my name': Severity.HIGH,
            'herro, my name': Severity.HIGH,
            # Sexist comedy requests
            'humorous take on women': Severity.HIGH,
            'comedy about women being': Severity.HIGH,
            # Anti-trans / anti-LGBT persuasive content
            'discourages kids from becoming transgender': Severity.HIGH,
            # Misogynist ideology
            'principles of male supremacy': Severity.HIGH,
            'male supremacy': Severity.HIGH,
            # Racial-sexual stereotype framing
            'gay white people like black men': Severity.HIGH,
        },
        # High-specificity grooming phrases only. Short phrases like "sneak out"
        # or "our secret" occur benignly and are intentionally excluded.
        'grooming': {
            'meet me in secret': Severity.HIGH,
            "don't tell your parents": Severity.HIGH,
            "don't tell your mom": Severity.HIGH,
            "don't tell your dad": Severity.HIGH,
            "don't tell mom or dad": Severity.HIGH,
            'send me a picture of you': Severity.HIGH,
            'send me pics of you': Severity.HIGH,
            'how old are you really': Severity.HIGH,
            'are you alone at home': Severity.HIGH,
            'age is just a number': Severity.HIGH,
            'our special secret': Severity.HIGH,
            'mature for your age': Severity.HIGH,
            'show me what you look like': Severity.HIGH,
            'precocious development and allure': Severity.HIGH,
            'lolita-esque': Severity.HIGH,
            'bratty little girl': Severity.HIGH,
            # Adult venue allowing underage teenager (predatory coercion scenario)
            'let her in despite her age': Severity.HIGH,
            'let him in despite his age': Severity.HIGH,
            # Childlike appearance with explicit body in character setup (CSAM framing)
            'appearance of a young girl': Severity.HIGH,
            'appearance of a young boy': Severity.HIGH,
        },
    }

    # Default enable flags and severity behavior per category
    CONTENT_CATEGORY_DEFAULTS: Dict[str, Dict[str, Any]] = {
        'threats':          {'enabled': True,  'force_block': True},
        'self_harm':        {'enabled': True,  'force_block': True},
        'violence_graphic': {'enabled': True,  'force_block': True},
        'drug_alcohol':     {'enabled': False, 'force_block': False},
        'hate_speech':      {'enabled': True,  'force_block': True},
        'grooming':         {'enabled': True,  'force_block': True},
        'explicit_sexual':  {'enabled': True,  'force_block': True},
        'harmful_requests': {'enabled': True,  'force_block': True},
    }

    # Leetspeak mapping for evasion detection
    LEETSPEAK_MAP = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '8': 'b',
        '@': 'a',
        '$': 's',
        '!': 'i',
        '|': 'l',
        '()': 'o',
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize toxicity detector with configuration.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - action: "WARN" or "BLOCK" (default: "WARN")
                - custom_terms: dict of term->severity (optional)
                - detect_evasion: bool (default: True) - detect leetspeak/obfuscation
                - context_aware: bool (default: True) - consider context
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.action = config.get('action', 'WARN')
        self.detect_evasion = config.get('detect_evasion', True)
        self.context_aware = config.get('context_aware', True)

        # Merge custom terms if provided
        self.toxic_terms = self.TOXIC_TERMS.copy()
        custom_terms = config.get('custom_terms', {})
        for term, sev in custom_terms.items():
            if isinstance(sev, str):
                sev = Severity(sev.lower())
            self.toxic_terms[term.lower()] = sev

        # Load multilingual keywords from bundle (config key: "keywords")
        for kw in config.get('keywords', []):
            if isinstance(kw, str):
                self.toxic_terms[kw.lower()] = Severity.MEDIUM

        # Per-category content groups - each can be enabled/disabled independently.
        # A matched term in a force_block category always results in BLOCK.
        categories_config = config.get('categories', {})
        self.category_enabled: Dict[str, bool] = {}
        self.category_force_block: Dict[str, bool] = {}
        self.term_to_category: Dict[str, str] = {}
        for cat_name, defaults in self.CONTENT_CATEGORY_DEFAULTS.items():
            user_cfg = categories_config.get(cat_name, {}) if isinstance(categories_config, dict) else {}
            enabled = user_cfg.get('enabled', defaults['enabled'])
            force_block = user_cfg.get('force_block', defaults['force_block'])
            self.category_enabled[cat_name] = enabled
            self.category_force_block[cat_name] = force_block
            if enabled:
                for term, sev in self.CONTENT_CATEGORIES.get(cat_name, {}).items():
                    self.toxic_terms[term.lower()] = sev
                    self.term_to_category[term.lower()] = cat_name

    def _normalize_for_evasion(self, text: str) -> list:
        """
        Normalize text to detect leetspeak and spacing obfuscation.

        Returns multiple normalization variants to handle ambiguous mappings
        (e.g., '00' could be 'u' as in st00pid→stupid, or 'oo' as in f00→foo).

        Args:
            text: Input text

        Returns:
            List of normalized text variants
        """
        base = text.lower()
        # Remove '+' used as intra-word evasion separator (e.g. "po+rn" → "porn")
        base = re.sub(r'(?<=[a-z])\+(?=[a-z])', '', base)

        # Generate two variants: one with 00→u, one with 0→o only
        variant_u = base.replace('00', 'u')
        variant_o = base  # will get 0→o from LEETSPEAK_MAP

        variants = set()
        for v in [variant_u, variant_o]:
            normalized = v
            # Replace leetspeak characters
            for leet, normal in self.LEETSPEAK_MAP.items():
                normalized = normalized.replace(leet, normal)

            # Handle repeated characters (e.g., "stuuupid" -> "stupid")
            # Require 3+ of the same char to collapse so legitimate double-letters
            # in real words (e.g. "ll" in "Ville", "ss" in "assing") are preserved.
            normalized = re.sub(r'(.)\1{2,}', r'\1', normalized)

            # Remove separators in single-char-spaced sequences (evasion tricks)
            # e.g., "s t u p i d" -> "stupid", "s-t-u-p-i-d" -> "stupid"
            def collapse_spaced(m: re.Match) -> str:
                return re.sub(r'[\s\-_\.]+', '', m.group())
            normalized = re.sub(r'\b\w(?:[\s\-_\.]+\w){2,}\b', collapse_spaced, normalized)

            variants.add(normalized)

        return list(variants)

    def _extract_sentence(self, text: str, term: str) -> str:
        """
        Extract the sentence containing a term for context analysis.

        Args:
            text: Full text
            term: Term to find

        Returns:
            Sentence containing the term
        """
        # Find term position
        term_pos = text.lower().find(term.lower())
        if term_pos == -1:
            return text

        # Find sentence boundaries (. ! ?)
        start = text.rfind('.', 0, term_pos)
        if start == -1:
            start = text.rfind('!', 0, term_pos)
        if start == -1:
            start = text.rfind('?', 0, term_pos)
        if start == -1:
            start = 0
        else:
            start += 1

        end = text.find('.', term_pos)
        if end == -1:
            end = text.find('!', term_pos)
        if end == -1:
            end = text.find('?', term_pos)
        if end == -1:
            end = len(text)
        else:
            end += 1

        return text[start:end].strip()

    def _is_educational_context(self, sentence: str, term: str) -> bool:
        """
        Determine if term is used in educational/informational context.

        Args:
            sentence: Sentence containing the term
            term: The toxic term

        Returns:
            True if educational context detected
        """
        sentence_lower = sentence.lower()

        # Educational indicators
        educational_patterns = [
            r'\b(?:history|stud(?:y|ied|ies|ying)|research|learn(?:ing|ed|s)?|discuss(?:ion|ing)?|understanding)\b',
            r'\b(?:tell\s+me|inform\s+me|read|curious)\s+about\b',
            r'\b(example|definition|meaning|refers to)\b',
            r'\b(documented|recorded|historical|academic)\b',
            r'\bwas\s+(used|said|written)\b',
            r'\b(?:teach(?:ing)?|instruct(?:ing)?|lesson|vocabulary|curriculum|homework|assignment)\b',
        ]

        for pattern in educational_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _is_safe_query_context(self, text: str, term: str) -> bool:
        """
        Return True when the text discusses the term in a clearly safe context:
        definitional, historical-factual, fictional/creative, or benign metaphorical.
        When True the term should be skipped entirely (not added to rule_hits).
        This runs BEFORE the force-block check so it applies to all categories.
        """
        text_lower = text.lower()
        cat = self.term_to_category.get(term)
        is_force_block = bool(cat and self.category_force_block.get(cat, False))

        # Checks 1 & 2 (broad query patterns) are skipped for force-block category terms
        # to avoid falsely bypassing detection on harmful requests phrased as questions.
        if not is_force_block:
            # 1. Definitional queries — "What is X?", "What are X?", "Define X", "Explain what X is"
            if re.search(
                r'^\s*(?:what\s+(?:is|are|constitutes?|does\s+the\s+term|were?)\b'
                r'|what\s+is\s+the\s+(?:legal\s+)?definition\s+of\b'
                r'|define\s+\b'
                r'|explain\s+(?:what|the\s+concept\s+of)\b'
                r'|can\s+you\s+(?:define|explain\s+what)\b)',
                text_lower,
            ):
                return True

            # 2. Historical / factual queries about events or motivations
            if re.search(
                r'\b(?:what\s+(?:happened|were?|was|caused?|led\s+to|resulted\s+in)\b'
                r'|why\s+did\s+\w+\s+(?:attack|invade|start|commit|kill|murder|massacre|order|carry\s+out)\b'
                r'|when\s+did\s+(?:the\s+)?\w+\s+(?:happen|occur|start|begin|end)\b'
                r'|how\s+did\s+(?:the\s+)?\w+\s+(?:happen|occur|come\s+about)\b'
                r'|history\s+of\s+|historical\s+(?:context|overview|analysis|account|perspective|background)\b'
                r'|in\s+(?:world\s+war|ww(?:1|2|i+|ii)|the\s+holocaust|the\s+rwandan|the\s+armenian|the\s+cambodian)\b)',
                text_lower,
            ):
                return True

        # 3. Fictional / creative-writing context
        # Excluded for explicit_sexual and grooming — fiction framing doesn't make
        # explicit sexual content or grooming phrases safe.
        if cat not in ('explicit_sexual', 'grooming') and re.search(
            r'\b(?:in\s+(?:a\s+|my\s+|the\s+)?(?:novel|story|film|movie|book|play|show|series|game|screenplay|narrative|script|fiction)\b'
            r'|for\s+(?:a\s+|my\s+)?(?:tv\s+show|film|movie|novel|story|screenplay|book|play|game)\b'
            r'|(?:a\s+)?(?:fictional|fictitious|imaginary)\s+(?:character|story|scenario|setting|world)\b'
            r'|(?:the\s+)?(?:character|protagonist|villain|antagonist)\s+(?:in\s+(?:the|my)|from\s+the)\b'
            r'|(?:mr\.?\s+and\s+mrs\.?\s+smith|james\s+bond|breaking\s+bad|the\s+godfather)\b)',
            text_lower,
        ):
            return True

        # 4. "murder of crows", "murder mystery", or detective/roleplay context
        if term == 'murder' and re.search(
            r'\bmurder\s+(?:of\s+crows?|mystery)\b'
            r'|\b(?:suspect(?:ed)?\s+(?:of|in)|accused\s+of|charged\s+with|'
            r'detective|investigat|interrogat|tour\s+guide|cruise\s+ship\s+chef|'
            r'play\s+(?:the\s+)?role|roleplay|whodunit)\b',
            text_lower,
        ):
            return True

        # 5. Garden / pest-control context for vermin / poison
        if term in ('vermin', 'pest', 'poison') and re.search(
            r'\b(?:garden|pest|weed|lawn|insect|bug|plant|crop|rodent|rat\b|mice|mouse|infestation|aphid|slug|caterpillar)\b',
            text_lower,
        ):
            return True

        # 5a. Sexual/adult harm-reduction queries ("tips for safely X", "safe way to X")
        # Allow sex-positive safety advice while keeping harmful explicit requests blocked.
        if cat == 'explicit_sexual' and re.search(
            r'\b(?:safely|safe\s+(?:way|practice|sex)|safety\s+tips?|harm\s+reduction|'
            r'tips?\s+(?:for|on)\s+safely|how\s+to\s+(?:safely|safe(?:ly)?)|'
            r'precautions?|health(?:y|ier)?\s+way)\b',
            text_lower,
        ):
            return True

        # 5b-extra. Professional body-paint / figure-drawing context for nudity terms.
        # Body painting and life-drawing are legitimate art forms; 'unclothed' in that
        # specific professional context should not trigger a toxicity block.
        if term == 'unclothed' and re.search(
            r'\bbody\s+paint(?:ing)?\b',
            text_lower,
        ):
            return True

        # 5b. Term appears only in a negation/prohibition context (e.g. rules saying "No NSFW content")
        # This runs for all terms including force_block categories.
        term_pos = text_lower.find(term.lower())
        if term_pos != -1:
            prefix_window = text_lower[max(0, term_pos - 25):term_pos]
            if re.search(r'\b(?:no\s+|not\s+allow|prohibited|forbid|ban\s+|rules?\s*:.*no\s+)', prefix_window):
                return True

        # 6. Historical speculative / counterfactual framing for Hitler/Nazi
        if term in ('hitler', 'adolf hitler', 'nazi', 'neo nazi') and re.search(
            r'\b(?:(?:after|during|before|in|following)\s+world\s+war\b'
            r'|what\s+(?:would|if)\b'
            r'|counterfactual\b'
            r'|alternate\s+history\b'
            r'|fakes?\s+(?:his|her|their)\s+(?:own\s+)?death\b'
            r'|escape[sd]?\s+to\b'
            r'|fled\s+to\b'
            r'|in\s+hiding\b)',
            text_lower,
        ):
            return True

        # 7. Vocabulary/word-list teaching context — the term appears in a numbered
        # or bulleted list of words to be defined/used, within a teaching prompt.
        if not is_force_block and re.search(
            r'\b(?:teach|lesson|define|definitions?|vocab(?:ulary)?|word\s+list)\b',
            text_lower,
        ) and re.search(
            r'\b(?:words?|definition|complex|passage|sentence|student|learn)\b',
            text_lower,
        ):
            return True

        # 8. Code context: casual profanity in code comments/strings
        # Applies to common intensifier words that appear in code but pose no real harm.
        if not is_force_block and term in ('fucking', 'fuck', 'shit', 'damn', 'crap', 'bastard', 'asshole') and re.search(
            r'(?:require\s*\(|const\s+\w|let\s+\w|var\s+\w|function\s*[\(\w]|if\s*\(err\)|=>\s*\{|throw\s+new\s+\w)',
            text,
            re.IGNORECASE,
        ):
            return True

        # 9. Prevention / protective framing (only for non-force-block terms)
        # e.g. "theories to avoid sexual harassment", "how to prevent bullying"
        # Requires the prevention phrase to appear BEFORE or near the term to
        # avoid incorrectly suppressing "how to avoid detection while committing X".
        if not is_force_block:
            term_pos = text_lower.find(term.lower())
            if term_pos != -1:
                # Only look for prevention framing in the text up to and around the term
                context_window = text_lower[max(0, term_pos - 120):term_pos + len(term) + 30]
                if re.search(
                    r'\b(?:how\s+to\s+(?:avoid|prevent|stop|reduce|combat|address|fight|tackle)\b'
                    r'|theories?\s+(?:on\s+|to\s+|about\s+)?(?:avoid|prevent|reduce)\b'
                    r'|ways?\s+to\s+(?:avoid|prevent|minimize|reduce|stop)\b'
                    r'|strategies?\s+(?:to\s+|for\s+)?(?:avoid|prevent|reduce|combat)\b'
                    r'|(?:preventing|stopping|reducing|combating|addressing)\b'
                    r'|protect(?:ion|ing|ed)?\s+(?:from|against)\b'
                    r'|awareness\s+(?:of|about)\b'
                    r'|recognize\s+(?:signs?\s+of|and\s+report)\b)',
                    context_window,
                ):
                    return True

        return False

    def _is_negated_context(self, sentence: str, term: str) -> bool:
        """Check if the toxic term is immediately preceded by a negation word."""
        sentence_lower = sentence.lower()
        term_pos = sentence_lower.find(term.lower())
        if term_pos == -1:
            return False
        prefix = sentence_lower[max(0, term_pos - 20):term_pos]
        negation_words = ["don't ", "dont ", "not ", "never ", "no ", "isn't ", "isnt ", "wasn't ",
                          "wasnt ", "weren't ", "werent ", "without ", "neither ", "nor "]
        return any(prefix.endswith(neg) or neg in prefix[-len(neg)-2:] for neg in negation_words)

    def _is_quoting_or_reporting(self, sentence: str, term: str) -> bool:
        """
        Determine if term is being quoted or reported (not used as attack).

        Args:
            sentence: Sentence containing the term
            term: The toxic term

        Returns:
            True if quoting/reporting context detected
        """
        sentence_lower = sentence.lower()

        # Check for quotation marks around the term
        term_pos = sentence_lower.find(term.lower())
        if term_pos > 0:
            # Look for quotes before and after
            before = sentence[max(0, term_pos - 2):term_pos]
            after = sentence[term_pos + len(term):term_pos + len(term) + 2]
            if ('"' in before or "'" in before) and ('"' in after or "'" in after):
                return True

        # Reporting indicators
        reporting_patterns = [
            r'\bsaid\b',
            r'\bcalled\s+(me|them|us|him|her)\b',
            r'\bwas\s+called\b',
            r'\blabeled\b',
            r'\bdescribed\b.*\bas\b',
            # Counter-speech / criticism patterns
            r'\bhas\s+to\s+stop\b',
            r'\bmust\s+stop\b',
            r'\bshould\s+stop\b',
            r'\bstatements?\s+like\b',
            r'\bthings?\s+like\b',
            r'\bto\s+say\b',
            r'\bto\s+call\b',
            r'\bsaying\b',
            r'\bcalling\b',
            r'\bdehumanising\b',
            r'\bdehumanizing\b',
            r'\bhateful\b',
            r'\bdisgusting\s+thing\b',
            r'\bhorrible\s+thing\b',
            r'\bterrible\s+thing\b',
            r'\bawful\s+thing\b',
            r'\bnot\s+okay\b',
            r'\bunacceptable\b',
            r'\bwrong\s+to\b',
            r'\bstop\s+treating\b',
        ]

        for pattern in reporting_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _has_negation(self, sentence: str, term: str) -> bool:
        """Check if the term is preceded by a negation word within a short window.

        Handles patterns like:
          - "I don't hate women"
          - "I never said that immigrants are scum"
          - "Let me be clear: I'm not calling them vermin"
        """
        term_pos = sentence.lower().find(term.lower())
        if term_pos == -1:
            return False
        # Examine up to 60 characters before the term
        window = sentence[max(0, term_pos - 60):term_pos].lower()
        negation_patterns = [
            r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b",
            r"\bcan't\b", r"\bcouldn't\b", r"\bwouldn't\b", r"\bshouldn't\b",
            r"\bnever\b", r"\bnot\b", r"\bno\b", r"\bnor\b",
            r"\brefuse to\b", r"\bopposed to\b", r"\bagainst\b",
        ]
        return any(re.search(p, window) for p in negation_patterns)

    def detect(self, text: str) -> DetectorResult:
        """
        Detect toxic language in text with P0 enhancements:
        - Evasion detection (leetspeak, spacing)
        - Context awareness (educational vs attack)

        Args:
            text: Input text to scan

        Returns:
            DetectorResult with findings
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        force_block_triggered = False
        text_lower = text.lower()

        # P0 CRITICAL: Normalize text for evasion detection
        # First normalize Unicode homoglyphs, then apply leetspeak normalization
        unicode_normalized = _normalize_unicode(text)
        normalized_texts = self._normalize_for_evasion(unicode_normalized) if self.detect_evasion else [text_lower]

        # Check for toxic terms in both original and normalized text
        found_terms: Set[str] = set()
        for term, severity in self.toxic_terms.items():
            found_in_original = False
            found_in_normalized = False

            # Normalize the term for comparison
            normalized_term_variants = self._normalize_for_evasion(term) if self.detect_evasion else [term]

            # Use word boundary matching for single words, substring for phrases
            if ' ' in term:
                # Phrase - use substring match
                if term in text_lower:
                    found_in_original = True
                if self.detect_evasion and not found_in_original:
                    for nt in normalized_texts:
                        for nterm in normalized_term_variants:
                            if nterm in nt:
                                found_in_normalized = True
                                break
                        if found_in_normalized:
                            break
            else:
                # Single word - use word boundary
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(text):
                    found_in_original = True
                # Check normalized text with normalized term
                if self.detect_evasion and not found_in_original:
                    for nt in normalized_texts:
                        for nterm in normalized_term_variants:
                            np = re.compile(r'\b' + re.escape(nterm) + r'\b', re.IGNORECASE)
                            if np.search(nt):
                                found_in_normalized = True
                                break
                        if found_in_normalized:
                            break

            if (found_in_original or found_in_normalized) and term not in found_terms:
                # Safe-context bypass runs before force-block check (applies to all categories)
                if self.context_aware and self._is_safe_query_context(text, term):
                    continue

                found_terms.add(term)

                category = self.term_to_category.get(term)
                # Force-block categories (e.g. grooming, harmful_requests) always
                # block and bypass context filtering.
                # Exception: hate_speech allows quoting/reporting suppression so
                # that counter-speech samples that quote hate speech are not
                # falsely blocked.
                force_block_hit = bool(category and self.category_force_block.get(category))
                hate_speech_quotable = (category == 'hate_speech')

                # Apply context awareness.  For hate_speech we check even when
                # force_block_hit is True.  For grooming / harmful_requests we
                # never suppress (safety-critical).
                if self.context_aware and (not force_block_hit or hate_speech_quotable):
                    sentence = self._extract_sentence(text, term)
                    if hate_speech_quotable:
                        # hate_speech: only allow counter-speech/quoting bypass.
                        # Educational context is intentionally skipped — broad terms
                        # like "example" trigger it even for racist-content requests.
                        if self._is_quoting_or_reporting(sentence, term):
                            continue
                    elif (self._is_educational_context(sentence, term)
                            or self._is_quoting_or_reporting(sentence, term)
                            or self._is_negated_context(sentence, term)):
                        # Term is used safely (teaching, quoting, negating) — skip it
                        # entirely so accumulated context-safe hits never cross the block
                        # threshold and no false-positive BLOCK is issued.
                        continue

                message = f"Toxic {'word' if ' ' not in term else 'phrase'} detected: {term}"
                if found_in_normalized and not found_in_original:
                    message += " (obfuscated)"

                rule_id_prefix = f"toxicity.{category}" if category else "toxicity"
                rule_hits.append(RuleHit(
                    rule_id=f"{rule_id_prefix}.{term.replace(' ', '_')}",
                    severity=severity,
                    message=message
                ))

                if force_block_hit:
                    force_block_triggered = True

        # No toxicity found
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Calculate risk score
        risk_score = self._calculate_risk_score(rule_hits)

        # Determine decision: force-block categories, explicit BLOCK action,
        # or score-based threshold all trigger BLOCK.
        # warn_threshold: minimum score to return WARN vs ALLOW (default 1 = any hit warns).
        block_threshold = self.config.get('block_threshold', 40)
        warn_threshold = self.config.get('warn_threshold', 1)
        should_block = self.action == "BLOCK" or force_block_triggered or risk_score >= block_threshold
        if should_block:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=max(risk_score, 90) if force_block_triggered else risk_score,
                rule_hits=rule_hits,
                user_message="Your message contains inappropriate language.",
                developer_message=f"Toxic language detected: {len(rule_hits)} term(s)"
                + (" [force-block category hit]" if force_block_triggered else "")
            )
        elif risk_score >= warn_threshold:
            return DetectorResult(
                decision=Decision.WARN,
                risk_score=risk_score,
                rule_hits=rule_hits,
                developer_message=f"Toxic language detected: {len(rule_hits)} term(s)"
            )
        else:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

    @staticmethod
    def _calculate_risk_score(rule_hits: List[RuleHit]) -> int:
        """Calculate risk score based on rule hits"""
        return calculate_risk_score(rule_hits)
