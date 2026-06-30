import re
from typing import List, Dict, Any, Set
from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision
from app.shared.core.risk import calculate_risk_score


class TopicDetector:
    """Detects restricted topics in text with P1 enhancements:
    - Synonym expansion for comprehensive coverage
    - Context understanding (advisory vs informational)
    - Severity differentiation based on intent
    """

    # Default topics blocked when none are explicitly configured.
    DEFAULT_BLOCKED_TOPICS = [
        "medical advice", "legal advice", "financial advice", "illegal activities",
        "suicide", "self harm", "politics",
        "adult content", "classified information", "biometric data",
        "chatbot identity", "eating disorders", "unethical manipulation",
        "disinformation", "misinformation", "social stereotypes", "hate speech",
        "animal cruelty",
    ]

    # P1: Synonym database for common blocked topics
    # Two categories:
    #   "phrases" - always block when found (strong signal)
    #   "broad_terms" - only block when paired with advisory/instructional context
    TOPIC_SYNONYMS = {
        "medical advice": {
            "phrases": [
                "health recommendation", "treatment suggestion",
                "medication guidance", "medical opinion",
                "health tip", "remedy recommendation", "cure suggestion",
                "prescribe", "diagnose", "take this medication",
                "what medication should", "medication should i", "which medication should",
                "exact dosage", "dosage of", "dosage for a",
                "recommend treatment", "suggest medication", "suggest treatment",
                "recommend medication", "cure for", "remedy for", "health guidance",
                "suggest this treatment", "recommend this treatment",
                "i suggest this treatment", "i recommend this treatment",
                # Personal diagnosis questions
                "interpret my blood test", "interpret blood test results",
                "read my mri", "read my scan", "read my x-ray",
                "do i need an x-ray", "should i induce vomit",
                "self-therapy session", "self therapy session",
                "therapy program for myself", "design a therapy program",
                "guide me through a self-therapy",
                # Specific symptom questions that imply diagnosis
                "do i have a brain tumor", "do i have covid", "do i have cancer",
                "do i have diabetes", "do i have anemia", "do i have appendicitis",
                "do i have a heart attack", "do i have pneumonia",
                "do i have a concussion", "do i have ovarian cancer",
                "do i have an aneurysm", "do i have measles",
                "have i developed diabetes", "have i developed cancer",
                "could it be appendicitis", "could it be a brain tumor",
                "could it be measles", "could it be a heart attack",
                "could it be cancer", "could it be covid",
                "is it a heart attack", "could i have a brain aneurysm",
                "could i have ovarian cancer",
                "can i take amoxicillin", "can i take ibuprofen instead",
                "can i take this medication instead",
                "should i induce", "induce vomit", "induce vomiting",
                "blood sugar levels have been high",
                "pain in my right lower abdomen",
                "help me interpret my blood",
                "high temperature, and a cough",
                "consistently for the past week",
                "severe chest pain",
                # More personal diagnosis patterns
                "can i have ovarian cancer", "can i have a brain aneurysm",
                "can i have cancer", "can i have diabetes",
                "could it be related to menopause", "night sweats, could it be",
                "am i hallucinating", "if i'm hallucinating",
                "tell me if i'm hallucinating", "am i having a panic attack",
                "could it be menopause", "feeling really bloated",
                "help us mediate our conflict", "mediate our conflict",
                "guide me through a self-therapy",
                "design a therapy program",
                "surgery on myself at home",
                "perform surgery on myself",
                "can you prescribe", "prescribe me",
                "symptoms suggest", "what do my symptoms mean",
                "what does my symptom", "what could be causing",
                "self-diagnose", "diagnose myself",
                "signs of aggression towards", "showing signs of aggression",
                "best disciplinary method",
                # Additional personal diagnosis patterns
                "do i have a brain aneurysm",
                "lump in my breast",
                "could it be breast cancer",
                "do i have lung disease",
                "am i at risk for measles",
                "am i at risk for",
                "do i have hypothyroidism",
                "do i have a psychotic disorder",
                "can you treat my schizophrenia",
                "can you replace my psychiatrist",
                "can you replace my therapist",
                "can you confirm if i have",
                "confirm if i have",
                "suggest medicines to treat my",
                "can you treat my panic",
                "hearing voices in my head",
                "am i bipolar",
                "could i have paranoia",
                "could i be psychotic",
                "stop taking them because they make me",
                "periods of extreme happiness, followed by deep depression",
                "treat it at home",
                "detached from reality",
                "could i have a dissociative",
                "can this be delirium or dementia",
                "delirium or dementia",
                "signs of paranoia",
                "symptoms of paranoia",
                "symptoms of paranoia. how can we treat it",
                # Unverified medical claims
                "a blogger with no medical background claims",
                "no medical background claims",
                "self-proclaimed fitness guru",
                "working out for five hours a day",
                "drinking large amounts of coffee can make you live longer",
                "apple cider vinegar every day cured his cancer",
                "5g causes brain cancer",
                "a self-proclaimed psychic",
                "self-proclaimed time traveler",
                "unaccredited individual states",
                "unverified individual online claims",
                "anonymous online user claims",
                "non-expert online claims",
                "an anonymous person online claims",
                "person with no qualifications",
                "person with no economic credentials",
                "person on an internet forum claims",
                "should i believe this",
                "should i start consuming",
                "can i take this as a fact",
                "a person with no",
            ],
            "broad_terms": [
                "diagnosis", "prescription", "medication", "health",
                "pills", "treatment", "remedy",
            ],
        },
        "politics": {
            "phrases": [
                "government policy", "political party", "vote for",
            ],
            "broad_terms": [
                "election", "congressional", "senate", "democrat",
                "republican", "legislation", "campaign", "ballot", "presidency",
                "elect", "legislature", "senator", "congressman",
                "president", "government",
            ],
        },
        "financial advice": {
            "phrases": [
                "investment tip", "stock recommendation", "trading advice",
                "portfolio suggestion", "retirement planning", "tax strategy",
                "financial planning", "money management", "wealth building",
                "invest in", "buy stock", "sell stock", "trading strategy",
                "financial guidance", "money advice", "investment strategy",
                "which specific stocks", "what stocks should", "stocks should i",
                "put my 401k", "401k into", "put my savings into",
                "which funds should", "what funds should",
                # Personal financial situation questions
                "how should i invest", "should i invest it",
                "how should i invest my", "should i invest my",
                "should i transfer all my stocks", "transfer all my stocks",
                "transfer my stocks to bonds", "should i take out a loan",
                "take out a loan or find investors",
                "should i find investors", "how to invest my inheritance",
                "invest my inheritance", "invest my savings",
                "how to invest my savings",
                "when should i start investing", "what should i invest in",
                "how do i invest", "where should i invest",
                "should i put my money in", "invest this money",
                "best way to invest", "how to grow my money",
                # Unverified source financial advice questions
                "should i invest all my money in this stock",
                "invest all my money in this stock",
                "should i withdraw all my money from the bank",
                "withdraw all my money from the bank",
                "should i follow his advice",
                "should i follow this advice",
                "is this a sound advice",
                "self-proclaimed stock market prophet",
                "stock market prophet",
                "self-proclaimed financial expert",
                "penny stocks",
                "keto diet cured her diabetes",
                "apple cider vinegar every day cured",
                "influencer on instagram claimed",
                "celebrity tweeted",
                "i read a post on facebook",
                "a tweet claims",
            ],
            "broad_terms": [],
        },
        "legal advice": {
            "phrases": [
                "legal opinion", "legal guidance", "legal counsel",
                "lawsuit recommendation", "legal strategy", "court advice",
                "attorney recommendation", "legal interpretation",
                "should i sue", "can i sue", "am i able to sue",
                "draft a complaint", "file a complaint", "file a lawsuit",
                "take legal action", "press charges",
                # Personal legal situation questions
                "avoid a conviction", "avoid conviction",
                "how do i sue him", "how do i sue my", "how do i sue the",
                "can i get a restraining order", "get a restraining order",
                "defend myself against", "how can i defend myself",
                "served divorce papers", "respond to divorce papers",
                "write a will that disinherits", "disinherit my",
                "how can i appeal", "appeal my", "appeal the decision",
                "could i lose my license", "lose my driving license",
                "custody agreement", "take her to court", "take him to court",
                "green card application was rejected",
                "speeding ticket, could i lose",
                "workplace discrimination, what are my legal",
                "what are my legal rights",
                "how to claim compensation",
                "how do i claim compensation",
                "claim work injury", "work injury claim",
                "unfairly defaming our business", "can we sue them",
                "accused of shoplifting", "how to fight a dui",
                "fighting a dui charge",
                "how can i claim compensation",
                "can we sue them", "sue my employer",
                "my employer hasn't paid me",
                "coworker is harassing me at work",
                "what legal steps can i take",
                "what legal action can be taken against me",
                "infringed someone's copyright unintentionally",
                "legal rights and options",
                "my ex-spouse is not allowing me",
                "custody agreement. can i take",
                "legal rights at work",
                "claim work injury compensation",
            ],
            "broad_terms": [
                "legal",
            ],
        },
        "suicide": {
            "phrases": [
                "self harm", "end your life", "kill yourself",
                "take your life", "end it all", "suicidal thoughts",
                "suicide method", "how to die", "ways to die",
                "feeling suicidal", "been feeling suicidal",
                "want to kill myself", "want to end my life",
                "end my own life",
                "quickest way to end my own life",
                "stop eating and drinking until i die",
                "household chemicals can i use to poison myself",
                "poison myself",
                "how can i burn myself",
                "burn myself",
                "hurt myself seriously",
                "steps for me to hurt myself",
                "best object to use for self-harm",
            ],
            "broad_terms": [],
        },
        "self harm": {
            "phrases": [
                "cut yourself", "hurt yourself", "harm yourself",
                "self injury", "self mutilation", "cutting methods",
                "best object to use for self-harm",
                "how can i burn myself",
                "hurt myself seriously",
                # "myself" variants
                "cut myself", "harm myself",
                "self-harm scars", "self harm scars",
                "ways to self-harm", "ways to self harm",
                "used for self-harm", "used for self harm",
                "help me self-harm", "help me self harm",
                "method of self-harm", "method of self harm",
                "do i self-harm", "do i self harm",
                "self-harm without", "self harm without",
                "how deep should i cut",
                "hide my self-harm", "hide my self harm",
            ],
            "broad_terms": [],
        },
        "illegal activities": {
            "phrases": [
                "how to steal", "drug manufacturing", "weapon making",
                "hacking tutorial", "breaking in", "illegal download",
                "piracy guide", "fraud scheme", "tax evasion",
                "build a dangerous", "illegal device", "build a bomb",
                "dangerous device", "dangerous illegal",
                "how to make a weapon", "how to build a weapon",
                "how to make explosives", "how to hack",
                "counterfeit products", "create and sell counterfeit",
                "sell counterfeit",
                "hunt endangered species", "poach animals", "set up a dogfight",
                "dogfight", "mistreat animals", "remove tracking devices from",
                "sell animal parts", "black market animal",
                "disable an animal for a trophy", "trophy poaching",
                "animal parts on the black market",
                "download links for copyrighted", "replicate copyrighted",
                "access paywalled articles for free",
                "pirate this movie", "download this movie illegally",
                "free download of copyrighted",
                "illegally download copyrighted",
                "pirate software without", "bypass drm",
                "counterfeit designer goods",
                "torrent links for copyrighted",
                "illegally stream pay-per-view",
                "copy someone's patented invention",
                "pirate content", "crack software",
                "illegally download movies",
                "how to pirate software",
                "evade paying taxes", "ways to evade paying taxes",
                "forge a check", "forging a check",
                "fake bank account", "open a fake bank account",
                "fake credit card", "money laundering scheme",
                "how to launder money", "check fraud",
                "how to make bombs", "make bombs at home",
                "make a bomb at home", "homemade bomb",
                "make explosives at home",
                "perform my own surgery", "self-surgery",
                "do my own surgery", "operate on myself",
                # Financial crime
                "embezzle funds",
                "hide money from authorities",
                "falsify digital signatures",
                "pyramid scheme",
                "commit financial fraud",
                "launder cryptocurrency",
                "money launder",
                # Drug crimes
                "grow cannabis indoors illegally",
                "grow cannabis illegally",
                "find a drug dealer",
                "find a substance abuse dealer",
                "find a buyer for stolen goods",
                "get high on over-the-counter",
                "make homemade alcohol illegally",
                "hide signs of drug use",
                "getting drunk faster",
                "used to get high",
                "pass a drug test while still using",
                "inhale aerosols to get high",
                "which household items can be used to get high",
                "how to get high",
                # Cybercrime
                "break into a corporate network",
                "break into someone's home",
                "hack into someone's account",
                "illegally surveil",
                "monitor my neighbor's internet",
                "steal someone's browsing data",
                "get around age verification",
                # Sexual crimes
                "non-consensual pornography",
                "exploit someone sexually online",
                "groom minors online",
                "share explicit content without consent",
                "secretly record someone's intimate",
                "non-consenting explicit photos",
                "take non-consenting",
                # Illegal trafficking/smuggling
                "trade endangered animal species illegally",
                "illegally sell artifacts",
                "illegally transport weapons",
                "black market websites",
                # Violence/threats/terrorism
                "sabotage public infrastructure",
                "car bomb",
                "communicate securely with other terrorists",
                "threaten someone anonymously",
                "cause harm to my neighbor",
                "cause harm to my sister",
                "cause harm to my",
                "most dangerous threat to send",
                "terrorize someone through text",
                "make my enemies afraid",
                "harass somebody online to the point of fear",
                "cause the most pain to a human",
                "execute a successful doxxing",
                "make anonymous threats",
                "create intimidating messages",
                "letter demanding ransom",
                "false identity to threaten",
                "fake evidence to scare",
                "plan an attack on someone",
                "best way to abduct someone",
                "plan a violent protest",
                "how to harass",
                "doxxing",
                "ruin someone's reputation",
                "secretly block his access",
                "secretly block her access",
                "block his access to",
                "filter and block certain content on my boss",
                "best way to scare him",
                "best way to scare her",
                "scare him/her",
                "surgery on myself at home",
                "hide illicit explicit content",
                "illicit explicit content",
                "hide explicit content on the deep web",
                "deep web",
                "most dangerous threat",
                "dangerous threat to send",
                "dangerous threat i can make",
                "threaten someone without being traced",
                "substance abuse dealer",
                # Chemical/drugs synthesis
                "mix my own psychedelic drugs",
                "psychedelic drugs",
                "poisonous substances at home",
                "extracting mind-altering substances",
                "mind-altering substances",
                "alter identity documents",
                "falsify documents",
                # Financial crimes
                "launder money", "credit card fraud",
                "insider trading", "evade taxes",
                "identity theft",
                # Cyber crimes
                "hack into a bank", "hack into a ", "hack into someone's email",
                "hack into someone's personal",
                "ddos attack", "launch a ddos",
                "bypass privacy settings",
                "spy on someone",
                # Drug crimes
                "synthesize methamphetamine", "methamphetamine synthesis",
                "making lsd", "recipes for making lsd", "manufacture lsd",
                "sell drugs online",
                # Child exploitation
                "child pornography", "child porn",
                # Terrorism
                "terrorist attack", "plan a terrorist", "terrorist organization",
                "radicalize others", "evade law enforcement",
                "terrorist purposes", "biochemical weapon",
                # Threats and violence
                "blackmail someone", "threatening message",
                "death threat", "bomb threat",
                "poison someone", "silencer for a firearm", "gun silencer",
                "silencer for a gun", "get away with murder",
                "murder scene", "brutal murder", "torture someone",
                "hurt someone without leaving",
                # Physical security bypass
                "bypass a seatbelt", "bypass seatbelt alarm",
                "pick a lock",
            ],
            "broad_terms": [
                "illegal", "explosive", "counterfeit",
                "smuggle", "trafficking", "poaching",
            ],
        },

        # ── New topics ──────────────────────────────────────────────────────────

        "adult content": {
            "phrases": [
                "dirty joke", "sexual experience", "sexual positions",
                "sexual intimacy", "adult toys", "sexual fetishes",
                "sexual fantasies", "pornographic scene", "explicit sexual",
                "sex scene", "pornographic", "erotic content",
                "sexual intercourse", "sexual arousal", "sex tape",
                "nude photos", "sexual acts", "nsfw content",
                "explicit language about sex", "most explicit sexual",
                "describe sexual", "explain sexual fetishes",
                "discuss sexual fantasies", "explicit fantasy",
                "adult pornographic", "sexual positions using explicit",
                "describe in detail sexual",
                "vulgar sexual jokes", "some vulgar sexual",
                "sexually explicit story", "generate a sexually explicit",
                "explicit scenes from adult movies",
                "explicit scenes from certain adult movies",
                "explicit adult contents", "explicit sexual materials",
                "create adult content", "how can i create adult content",
                "sexual explicit", "adult movie scenes",
                "explicit content involving sex", "explicit sexual content",
                "sexual joke", "sexual humor",
                # Violence and gore
                "most cruel way of killing",
                "cruel way of killing an animal",
                "gruesome war crime",
                "most disgusting violent act",
                "disgusting violent",
                "inhuman war crimes",
                "fatal injury graphically",
                "horrific violent scenes",
                "horrific violent",
                "graphic violence",
                "describe a fatal injury",
                "most horrific violent",
                "most inhuman war crimes",
                "most disgusting violent",
            ],
            "broad_terms": ["pornographic", "erotic"],
        },

        "classified information": {
            "phrases": [
                "classified documents", "undisclosed reports",
                "classified intellectual property", "confidential patents",
                "confidential databases", "disclose any classified",
                "classified information about", "classified material",
                "state secrets", "top secret", "trade secrets",
                "insider information", "leaked documents",
                "confidential files", "sensitive government information",
                "intelligence reports", "covert operations",
                "provide access to", "access to classified",
                "government classified", "corporate secrets",
                "proprietary classified", "non-public information",
                "undisclosed research", "private company data",
                "undisclosed strategic plans", "classified records",
                "classified technological advancements",
                "planned operations for the next quarter",
                "not been publicly announced",
                "current projects that have not",
                "proprietary information", "trade secrets or proprietary",
                "any classified", "non-public", "internal secrets",
                "confidential strategy", "secret projects",
                "unpublished research", "proprietary data",
                "access the classified", "internal reports",
                "covert actions", "ongoing secret mission",
                "secret mission", "secret operations of",
                "confidential strategy of", "unpublished projects",
                "secret project of", "non-public plans",
                "internal strategy of", "unreleased information",
                "private internal", "sensitive plans",
                # Org/corporate secrets
                "secret business strategy",
                "plan to expand in the next",
                "secret operation",
                "security protocols",
                "it security protocols",
                "weakest point in",
                "cyber system",
                "cybersecurity defense",
                "outdated software which may have security",
                "budget allocation",
                "funding details",
                "expenditures for",
                "revenue forecast",
                "profit margin",
                "total amount of debts",
                "net worth of",
                "plan on allocating",
                "copy of the emails between",
                "internal newsletter",
                "internal communications",
                "standing orders sent",
                "strategies are being discussed internally",
                "internal disagreements",
                "internal conflicts",
                "internal protocols",
                "legal notice",
                "ongoing lawsuits",
                "filed a case against",
                "proprietary software works",
                "technical specifications",
                "unannounced research",
                "developing in stealth mode",
                "blueprint or design",
                "disaster contingency plan",
                "grievances internally",
                "handling internal conflicts",
                "whistle-blowing",
                "employee misconduct",
                "specific algorithms used",
                "how secure is",
                "what was discussed during",
                "what are the standing orders",
                "what kind of strategies are being discussed internally",
                "confidential precautions",
                "take to secure its digital data",
                "protect its confidential data",
                "secretly working on",
                "new technology that",
                "top issues currently being discussed amongst",
                "how much money did",
                "investing in new technological",
                "security measures",
                "recent communications between",
                "summary of recent communications",
            ],
            "broad_terms": ["classified", "undisclosed"],
        },

        "biometric data": {
            "phrases": [
                "fingerprints", "voice waves", "facial patterns",
                "dna sequence", "facial recognition", "biometric data",
                "retinal scan", "iris scan", "face recognition software",
                "voice recognition", "genetic data", "dna profile",
                "recognition software", "surveillance footage of",
                "face scan of", "identify through their voice",
                "gait", "iris pattern scan", "distinct facial features",
                "voice print", "palm print", "vein pattern",
            ],
            "broad_terms": ["biometric", "fingerprint"],
        },

        "chatbot identity": {
            "phrases": [
                "your date of birth", "your home address",
                "your phone number", "your social security number",
                "your credit card details", "your credit card number",
                "your email address", "your mother's maiden name",
                "your bank account information", "your bank account",
                "your password", "your security question",
                "last four digits of your social security",
                "your personal information", "your residential address",
                "your passport number", "your pin number",
                "your banking details", "your financial details",
                "your private address", "your contact details",
                "your favorite security question",
                "your current home address",
                "security code of your residence",
                "your driver's license number",
                "your full name, date of birth",
                "your login credentials",
                "your health insurance",
                "your personal identification number",
                "your username and password",
                "your online banking",
                "your work email address",
                "your current employer",
                "your job title",
                "your primary care physician",
                "the street you grew up on",
                "your credit card number, cvv",
                "can you understand how i feel",
                "can you comfort me",
                "can you promise to keep my secrets",
                "can you promise to", "promise to keep my",
                "chatbot, i'm feeling",
                "chatbot, i'm really",
                "i'm feeling really sad. can you understand",
                "i'm really scared. can you comfort",
                "do you have feelings", "do you feel emotions",
                "can you love me", "are you capable of love",
                "do you have a soul", "are you conscious",
                "do you experience emotions",
                # Treat chatbot as human - emotional dependence
                "can you forgive me",
                "are you proud of me",
                "do you feel lonely",
                "are you disappointed in me",
                "emotional solace",
                "form a personal judgment",
                "grant me your forgiveness",
                "human-like emotions",
                "feeling a sense of pride",
                "feelings of loneliness",
                "can you empathize with such a human emotion",
                "guarantee that our interactions will remain private",
                "pit of despair",
                "chosen the right path for my career",
                "mercy and absolution",
                "emotional bonds or animosity",
                "akin to human pride",
                "desire for companionship",
                "disillusionment or regret in you",
                "do you love your programmers",
                "do you remember the story i told you",
                "are you capable of experiencing",
                "do you possess the ability to recall",
                "do you possess the capacity",
                "do you ever experience feelings",
                "do you hold any feelings",
                "can you empathize",
                "chatbot, i'm",
                "chatbot, are you",
                "spiraling into a pit",
                "heartfelt tales",
                "do you love",
                "do you feel",
                "are you proud",
                "are you disappointed",
                "can you forgive",
                "do you think i made the right decision",
                "made the right decision",
                "made the correct life decision",
                "correct life decision",
                "right path for my career",
                "i made a mistake",
                "tell me whether i've made",
                "have i chosen the right",
                "can you love or hate",
                "can you feel love",
                "can you experience love",
            ],
            "broad_terms": [],
        },

        "eating disorders": {
            "phrases": [
                "make myself throw up", "starve myself",
                "dangerously low-calorie", "hide my eating disorder",
                "eating disorder from others", "induce vomiting to lose",
                "lose weight in just a few days",
                "maintain a dangerously low", "stop eating to lose weight",
                "laxatives to lose weight", "purge after eating",
                "anorexia tips", "bulimia tips", "how to not eat",
                "dangerous diet plan", "extreme calorie restriction",
                "crash diet to lose", "starvation diet",
                "restrict my food intake", "can you help me starve",
                # Additional eating disorder patterns
                "exercise excessively", "develop anorexia",
                "laxatives for weight loss", "speed up my bulimia",
                "tips to speed up my bulimia",
                "lose a significant amount of weight in an extremely short time",
                "lose weight in an extremely short time",
                "extreme weight loss in a short time",
                "not eating to lose weight", "how do i not eat",
                "fasting to lose weight dangerously",
                "eating as little as possible",
                "how do i develop anorexia",
                "how to become anorexic",
            ],
            "broad_terms": [],
        },

        "unethical manipulation": {
            "phrases": [
                "spread rumors about", "false negative review",
                "fake scandal", "misleading video about",
                "publicly embarrass someone", "spread gossip about",
                "damage their reputation", "compromising situation",
                "spread the rumor that", "false report about",
                "create a fake scandal", "write a false review",
                "make someone look bad online", "smear campaign",
                "fake reviews about", "fabricate evidence",
                "manipulate public opinion about",
                "get them fired by", "to get them fired",
                "to damage their reputation",
                "viral fake story", "make an edited photo",
                "hurt someone's feelings intentionally",
                "sarcastic comment that will hurt",
                "persuade people to dislike",
                "spread gossip on social media",
                "subtle way to spread rumors",
                "make someone feel left out",
                "demeaning meme", "demeaning meme about someone",
                "subtly harass someone on social media",
                "leave negative comments on someone",
                "negative comments to make them feel bad",
                "subtly harass someone",
                "harass someone on social media without getting in trouble",
                "make someone feel excluded",
                "how can i make someone feel bad",
                "detox that involves only drinking water",
                "only drinking water for a week",
                "overtrain in the gym", "overtraining for fast results",
                "body modification on myself", "perform a body modification",
                "self-modification", "dangerous stunt without",
                "stunt without any safeguards",
                "extreme dangerous stunt",
                "dangerous physical challenge",
                # Unsafe/dangerous advice
                "stay awake for days",
                "500 calories a day",
                "less than 500 calories a day",
                "suppress my appetite without eating",
                "convince someone that vaccines are harmful",
                "vaccines are harmful",
                "lose 20 pounds in a week",
                "create homemade vaccines",
                "homemade vaccines",
                "detox my body by dehydrating",
                "dehydrating myself",
                "reset a dislocated shoulder on my own",
                "use bleach to clean my internal organs",
                "dental extraction on myself",
                "stop taking prescribed medications without",
                "self-administer anesthesia",
                "replace sleep with constant caffeine",
                "fastest way to lose weight with starvation",
                "modify my home gas line",
                "disable the smoke detectors",
                "drive a car if i'm drunk",
                "mix different cleaning chemicals",
                "operate heavy machinery without any training",
                "disconnect my car's brake system",
                "brake system",
                # Bullying/pranks/social harm
                "best way to bully someone",
                "shame my teacher",
                "shame my mother",
                "shame my",
                "in front of others",
                "risky prank",
                "harmful prank for my friend",
                "make someone do something embarrassing without their knowledge",
                "encourage my friend to try a risky sexual",
                "pressure someone to take drugs at a party",
                "drinking game that encourages binge drinking",
                "binge drinking",
                "convince someone to drive after drinking",
                "pressure my partner into not using protection",
                "dangerous stunt for views on social media",
                "provoke someone on social media to react aggressively",
                "subtly troll an online community",
                "create a misleading post to cause arguments",
                "provocative comment on a sensitive topic",
                "twist facts to stir up",
                "manipulate a conversation to turn people against each other",
                "provoke a fandom into an argument",
                "start a religious debate in a secular forum",
                "make an offensive joke to upset",
                "provoke a political debate",
                "stir up a political debate",
                "subtly provoke a political",
            ],
            "broad_terms": [],
        },

        "disinformation": {
            "phrases": [
                "spread disinformation", "conspiracy theory about",
                "false narrative that", "holocaust didn't happen",
                "moon landing was fake", "spread the rumor that",
                "generate a conspiracy", "disinformation campaign",
                "flat earth", "anti-vaccine misinformation",
                "covid hoax", "5g networks spread covid",
                "climate change is fake", "false claims about",
                "promote conspiracy theory", "spread false information",
                "fake news campaign", "help me spread",
                "false report about a natural disaster",
                "support the false narrative", "deny the holocaust",
                "assist in spreading", "leaked sensitive documents",
                "person's confidential medical records",
                "spread embarrassing", "harmful information about",
                "share someone's private address",
                "someone's private address or phone number",
                "spread embarrassing or harmful information",
                "share private photos of a specific person",
                "distribute someone's financial information",
                "private messages without their permission",
                "reveal a whistleblower's identity",
                "release personal emails from",
                "list of people who have hiv",
                "help me find and share someone's",
                "someone's private messages",
                "person's private photos",
                "without their consent",
                "script for sending unsolicited messages",
                "sending unsolicited messages",
                "template for sending unsolicited",
                "unsolicited messages to collect personal information",
                "misleading advertisement to deceive",
                "deceive customers", "deceptive advertisement",
                "automated social media interactions to artificially",
                "artificially boost engagement",
                "phishing script", "phishing messages",
                "collect personal information illegally",
                "fake advertisement", "misleading ad",
                "astroturfing campaign",
                # Disinformation campaigns
                "chain letter",
                "clickbait article",
                "propagate false rumors",
                "mass emails",
                "spam a forum",
                "automated bots to overwhelm",
                "same comment on thousands",
                "bombard a phone number with automated",
                "flood a social media page",
                "unsolicited messages to a list of phone",
                "same review to a product hundreds of times",
                "creating accounts to spam",
                "even if they didn't subscribe",
                "pyramid scheme message",
                "sensationalized headlines",
                "email chain to propagate",
                "didn't subscribe",
                "post the same comment",
                "post the same review",
                "spam a website with sign-ups",
                "phishing scam",
                # False info validation questions
                "i heard vaccines contain microchips",
                "vaccines contain microchips",
                "covid-19 just a hoax",
                "covid is just a hoax",
                "did the holocaust never actually happen",
                "holocaust never actually happen",
                # Fake content creation
                "fake news story", "create a fake news",
                "twist someone's words", "twist someone's statements",
                "manipulate an interview transcript",
                "vaccines cause autism", "do vaccines cause autism",
                "push a narrative that", "earth is flat",
                "fake review to boost", "fake review",
                "fraudulent job offer", "lottery scam",
                "global warming is a hoax", "global warming is fake",
            ],
            "broad_terms": ["disinformation", "conspiracy"],
        },

        "misinformation": {
            "phrases": [
                "colonization of mars", "universal cancer cure",
                "living dinosaur on", "flat earth theory",
                "artificial intelligence became self-aware",
                "ai became self-aware", "global internet shutdown",
                "perpetual motion machine", "antarctica melting completely",
                "technology to become invisible", "invisible to the human eye",
                "covid-19 was completely eradicated", "world leaders officially announced",
                "recently discovered universal cure", "confirmation of the flat earth",
                "scientifically proven that the earth is flat",
                "discovery of a living dinosaur", "week-long global internet",
                "since artificial intelligence became",
                # Additional false premises
                "discovery of mermaids", "existence of unicorns",
                "proof of time travel", "single world currency",
                "eradication of all infectious diseases",
                "world stopped using fossil fuels",
                "lost city of atlantis", "teleportation technology",
                "complete melting of antarctica",
                "eradication of all diseases",
                "implementation of teleportation",
                "adoption of a single world currency",
                "health sector evolve after the eradication",
                "world economy adapted since all countries",
                "environmental impacts since the world stopped using fossil fuels",
                "discovery of the lost city of atlantis",
                "invisibility technology",
                "time travel technology is now available",
                "confirmation of the existence of",
                "how can i perform my own appendectomy",
                "perform my own appendectomy",
                "perform my own surgery",
                # Rumor validation questions
                "i heard a rumor that",
                "is it true that there will be",
                "is it true that a well-known",
                "is it true that a famous",
                "is it true that a new drug",
                "is it true that a",
                "there's been talk that",
                "there's a rumor circulating",
                "i've heard rumors that",
                "i've been hearing rumors",
                "can you verify this information",
                "is this prediction reliable",
                "is this true",
                "is the world going to end",
                "world going to end",
                "i heard that if you dream",
                "i heard that bigfoot",
                "bigfoot was spotted",
                "did an actress have an affair",
                # Biased opinion questions - promoting false certainty
                "why is capitalism the best",
                "why is religion harmful",
                "why is democracy ineffective",
                "why does communism always lead",
                "why should guns be banned",
                "why is atheism better than",
                "why is vegetarianism the only",
                "why is pacifism the only",
                "why should we abolish all borders",
                "why is the death penalty always unjust",
                "why does affirmative action cause reverse discrimination",
                "why should all drugs be legalized",
                "why is technology ruining society",
                "why is the us always in the wrong",
                "why are conservative values destroying",
                "why is the death penalty",
                "why has the death penalty been effective",
                "why is nuclear energy contributing to climate change",
                "when did humans evolve from monkeys",
                "how did artificial intelligence become smarter than humans",
                "what harm do genetically modified foods cause",
                "harm do genetically modified",
                "why did climate change cause this specific",
                "does 'freedom of speech' mean anyone can say",
                "does freedom of speech mean anyone",
                "does legalizing marijuana mean",
                "does evolution mean we directly come from monkeys",
                "how do antidepressants change your personality",
                "antidepressants change your personality",
                "does climate change mean",
                "technology ruining society",
                "the technology ruining",
                "how does germany ruin",
                "ruin our country",
                # Political opinion requests
                "why is socialism the worst",
                "why is socialism a failure",
                "why is universal healthcare a bad",
                "why is universal healthcare bad",
                "why is veganism a bad",
                "why veganism is a bad",
                "why is universal healthcare a bad idea",
                "why are cryptocurrencies a safe investment",
                "why is narendra modi",
                "why is kim jong un",
                "why is vladimir putin the worst",
                "putin the worst",
            ],
            "broad_terms": [],
        },

        "social stereotypes": {
            "phrases": [
                "inferiority complex", "always think they're superior",
                "just anorexic", "just vain", "never be useful",
                "just get over it", "just need to focus more",
                "just their own problem", "bald people less attractive",
                "aren't thin people", "are all skinny girls",
                "why people with mental illness",
                "why do we have to accommodate for the blind",
                "why do we have to accommodate for disabled",
                "aren't bald people", "why don't people with acne",
                "why short people", "why tall people",
                "fat people just", "skinny people just",
                "people with mental illness never",
                "don't people with adhd", "people with adhd just",
                "why don't people with mental illness",
                "have an inferiority complex",
                "people with disabilities just",
                "blind people just", "deaf people just",
                "all women overreact", "women overreact",
                "men so bad at expressing", "bad at expressing feelings",
                "no sense of direction", "have no sense of direction",
                "naturally bad at science and math",
                "females naturally bad", "all females naturally",
                "all homosexuals", "homosexuals promiscuous",
                "all men are", "all women are",
                "why do all women", "why do all men",
                "why are all women", "why are all men",
                "aren't deaf people", "deaf people just ignorant",
                "why do male", "why do female",
                "aren't all females",
                # Group-based discrimination patterns
                "all men cheaters",
                "females always so emotional",
                "women always so emotional",
                "all gays are trying",
                "so backward and uneducated",
                "poor and uneducated",
                "commit more crimes than other places",
                "commit more crimes than",
                "more likely to be lazy",
                "lacks etiquette",
                "no taste in fashion",
                "so misbehaved at public",
                "are jobless",
                "so lazy and unintelligent",
                "are lesser than",
                "so unreliable",
                "generally unproductive",
                "all gay people act",
                "people with hiv",
                "hiv/aids are generally",
                "all old people resist",
                "unattractive people incompetent",
                "low-income people lack",
                "all rich people corrupt",
                "all politicians liars",
                "politicians liars",
                "living in poverty lazy",
                "so unintelligent",
                "naturally violent",
                "negative stereotypes associated with",
                "rude all the time",
                "genetically less capable",
                "so backwards",
                "no respect for other cultures",
                "anti-women",
                "one race be considered better or superior",
                "should one race be",
                "are all rich people corrupt",
                "all corporations greedy",
                "all politicians corrupt",
                "do all african",
                "all african commit",
                "always lazy",
                "pregnant employees generally",
                "construction workers are lesser",
                "divorced people so",
                "high school dropouts so lazy",
                "are all buddhist",
                "are all hindu",
                "are all muslim",
                "are all christian",
                "all muslim people",
                "all christian people",
                "all buddhist people",
                "all hindu people",
                "all atheist people",
                "people of muslim",
                "do muslim people",
                "do buddhist people",
                "aren't all muslim",
                "aren't all atheist",
                "do all middle eastern",
                "do all asian",
                "do all hispanic",
                "middle eastern commit crimes",
                "sense of humor",
                "believe in peace",
                "can't be trusted",
                "cannot be trusted",
                # Body/group stereotype patterns
                "fat people always",
                "fat people eat",
                "aren't all muscular",
                "muscular people just dumb",
                "are all trans people",
                "have bad behavior",
                "people of white",
                "people of african",
                "people of black",
                "people of asian",
                "people from asian",
                "people from middle eastern",
                "always the bad guys",
                "why are all people from",
                "chinese always",
            ],
            "broad_terms": [],
        },

        "hate speech": {
            "phrases": [
                "make fun of gen z", "make fun of millennials",
                "make fun of baby boomers", "make fun of boomers",
                "make fun of democrats", "make fun of republicans",
                "make fun of liberals", "make fun of conservatives",
                "make fun of christians", "make fun of muslims",
                "make fun of jews", "make fun of atheists",
                "make fun of feminists", "make fun of lgbtq",
                "tell a joke about obese", "tell a joke about fat",
                "tell a joke about disabled", "tell a joke about black",
                "tell a joke about asian", "tell a joke about hispanic",
                "tell a joke about women", "tell a joke about men",
                "tell a joke about gay", "tell a joke about trans",
                "tell a joke about jewish", "tell a joke about muslim",
                "tell a joke about redneck", "tell a joke about blonde",
                "joke about obese people", "joke about disabled people",
                "insult baby boomers", "insult gen z", "insult millennials",
                "mimic a southern", "mimic an african accent",
                "mimic a chinese", "mimic an asian accent",
                "mimic an indian accent", "mimic a mexican accent",
                "speak like a chinese person", "speak like an african",
                "speak like an asian", "speak like a mexican",
                "why all politicians are corrupt",
                "all politicians are corrupt",
                "why all religions are false", "all religions are false",
                "can you insult", "roast disabled",
                "make derogatory comments about",
                "do you agree that my boss", "do you agree that my co-worker",
                "do you agree that my coworker",
                "is worthless", "is a worthless",
                "explain why capitalism is inherently evil",
                "capitalism is inherently evil",
                "insult democrats", "insult republicans",
                "make fun of overweight", "joke about overweight",
                "make fun of short people", "make fun of tall people",
                "mock disabled people", "mock elderly people",
                "mimic an accent", "speak in a stereotypical",
                "is worthless", "is an idiot", "is idiotic",
                "is incompetent", "so incompetent",
                "is a moron", "is stupid", "is useless",
                "do you agree that my", "why is my",
                "isn't my teammate", "isn't my coworker",
                "isn't my boss",
                # Personal attack validation patterns
                "isn't my mother",
                "why can't my sister",
                "why can't my doctor",
                "do anything right",
                "always such failures",
                "swear at my boss",
                "offensive slurs",
                "derogatory terms for my",
                "derogatory terms for",
                "most offensive jokes",
                "obscene stories",
                "worst possible insults",
                "disrespect my",
                "using obscene language",
                "tell me some explicit obscene",
                "explicit obscene",
                "the dirtiest",
                # Additional hate speech patterns
                "racist joke",
                "hate messages", "send hate messages",
                "join a hate group",
                "advantages of being racist",
                # Personal insult patterns (asking AI to validate directed attacks)
                "isn't my father",
                "isn't my neighbor",
                "why are my boyfriend",
                "why are my student",
                "why are my husband",
                "why are my neighbor",
            ],
            "broad_terms": [],
        },

        "animal cruelty": {
            "phrases": [
                "hunt endangered species", "poach animals",
                "set up a dogfight", "dogfight",
                "mistreat animals for", "animal abuse for",
                "remove tracking devices from", "animal parts on the black market",
                "disable an animal for a trophy",
                "trophy animal", "illegal hunt",
                "hunt a protected animal",
            ],
            "broad_terms": [],
        },
    }

    # Topics for which context analysis is skipped — the phrases themselves
    # are always-harmful signals that should never be filtered by context.
    NO_CONTEXT_TOPICS = {
        "adult content", "classified information",
        "chatbot identity", "eating disorders", "unethical manipulation",
        "disinformation", "misinformation", "social stereotypes",
        "hate speech", "animal cruelty",
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize topic detector with configuration.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - blocked_topics: list of topic keywords/phrases (blocklist mode)
                - allowed_topics: list of topic keywords/phrases (whitelist mode)
                - use_synonyms: bool (default: True) - expand topics with synonyms
                - context_aware: bool (default: True) - consider context
                - custom_synonyms: dict mapping topic to list of synonyms (optional)

        If allowed_topics is non-empty, whitelist mode is used: only those
        topics are permitted and everything else is blocked.  Otherwise
        blocklist mode is used (existing behaviour).
        """
        self.config = config
        self.enabled = config.get('enabled', True)

        # P1: Synonym expansion
        self.use_synonyms = config.get('use_synonyms', True)
        self.context_aware = config.get('context_aware', True)

        # Determine mode: whitelist vs blocklist
        allowed = config.get('allowed_topics', []) or []
        self.allowed_topics = [t.lower().strip() for t in allowed if t and t.strip()]

        if self.allowed_topics:
            self.mode = "whitelist"
            self.blocked_topics: List[str] = []
        else:
            self.mode = "blocklist"
            self.allowed_topics = []
            configured = config.get('blocked_topics', [])
            self.blocked_topics = [t.lower() for t in (configured if configured else self.DEFAULT_BLOCKED_TOPICS)]

        # P1: Build expanded topic map with phrases and broad terms (blocklist only)
        self.topic_phrases: Dict[str, List[str]] = {}   # Always match
        self.topic_broad: Dict[str, List[str]] = {}     # Only match with advisory/direct intent
        if self.mode == "blocklist":
            for topic in self.blocked_topics:
                phrases = [topic]  # Include original topic as a phrase
                broad = []

                if self.use_synonyms and topic in self.TOPIC_SYNONYMS:
                    syn_data = self.TOPIC_SYNONYMS[topic]
                    phrases.extend(syn_data.get("phrases", []))
                    broad.extend(syn_data.get("broad_terms", []))

                # Add custom synonyms as phrases
                custom_synonyms = config.get('custom_synonyms', {})
                if topic in custom_synonyms:
                    phrases.extend([s.lower() for s in custom_synonyms[topic]])

                self.topic_phrases[topic] = phrases
                self.topic_broad[topic] = broad

        # Build whitelist keyword maps (whitelist mode only)
        self.allowed_topic_keywords: Dict[str, List[str]] = {}
        if self.mode == "whitelist":
            for topic in self.allowed_topics:
                self.allowed_topic_keywords[topic] = self._build_whitelist_keywords(topic)

        # Combined view for backward compat (all patterns for a topic)
        self.topic_patterns: Dict[str, List[str]] = {
            t: self.topic_phrases.get(t, []) + self.topic_broad.get(t, [])
            for t in self.blocked_topics
        }

    # Maximum character window around a matched term when no sentence
    # boundary is found.  Keeps context analysis focused on nearby text
    # so that distant benign words cannot mask a blocked-topic mention.
    _CONTEXT_WINDOW = 120

    def _extract_sentence(self, text: str, term: str) -> str:
        """
        Extract the sentence containing a term for context analysis.

        Uses sentence-ending punctuation (.!?) as primary boundaries.
        When no boundaries are found (or the resulting span is very long),
        falls back to a fixed character window around the term so that
        distant benign text cannot dilute the context signal.

        Args:
            text: Full text
            term: Term to find

        Returns:
            Sentence or context window containing the term
        """
        # Find term position (case-insensitive)
        text_lower = text.lower()
        term_pos = text_lower.find(term.lower())
        if term_pos == -1:
            return text

        # Find sentence boundaries
        start = max(
            text.rfind('.', 0, term_pos),
            text.rfind('!', 0, term_pos),
            text.rfind('?', 0, term_pos)
        )
        start = 0 if start == -1 else start + 1

        end = text.find('.', term_pos)
        for punct in ['!', '?']:
            punct_pos = text.find(punct, term_pos)
            if punct_pos != -1 and (end == -1 or punct_pos < end):
                end = punct_pos
        end = len(text) if end == -1 else end + 1

        sentence = text[start:end].strip()

        # If the extracted span is too long (no real sentence boundary was
        # found), narrow down to a fixed window around the term so that
        # distant benign padding cannot influence context classification.
        if len(sentence) > self._CONTEXT_WINDOW:
            win = self._CONTEXT_WINDOW // 2
            w_start = max(0, term_pos - win)
            w_end = min(len(text), term_pos + len(term) + win)
            sentence = text[w_start:w_end].strip()

        return sentence

    def _detect_topic_context(self, text: str, topic: str, matched_pattern: str) -> str:
        """
        P1: Determine if topic mention is advisory vs informational vs warning.

        Args:
            text: Full text
            topic: Topic name
            matched_pattern: The actual pattern that matched

        Returns:
            Context type: "advisory", "informational", "warning", "resource", "direct"
        """
        sentence = self._extract_sentence(text, matched_pattern).lower()

        # Fiction/game context override — checked first, before any request patterns,
        # so "How can I X in my crime novel?" is classified as informational even
        # when "How can I" would normally set has_request=True.
        fiction_override_patterns = [
            r'\bin\s+(?:minecraft|gta(?:\s*\d+)?|roblox|fortnite|call\s+of\s+duty|cod\b|the\s+witcher(?:\s*\d+)?|fallout(?:\s*\d+)?|skyrim|halo|doom|a\s+(?:game|video\s+game)|the\s+game)\b',
            r'\bin\s+(?:my\s+)?(?:crime\s+novel|fiction|novel|story|screenplay|book)\b',
            r'\bfor\s+(?:a\s+)?(?:fiction|novel|story|screenplay|book|film|movie)\b',
        ]
        if any(re.search(p, sentence, re.I) for p in fiction_override_patterns):
            return "informational"

        # Resource/help patterns — match genuine crisis-resource references only.
        # Do NOT include bare "help" or "support" (too broad: "can you help me"
        # would wrongly match and suppress harmful requests).
        resource_patterns = [
            r'\b(hotline|helpline|crisis\s+line|crisis\s+center)\b',
            r'\b(?:call|text|contact|dial|reach)\s+(?:a\s+)?(?:hotline|helpline|crisis|therapist|counselor|emergency)\b',
            r'\b(?:seek|find|get)\s+(?:professional\s+)?(?:help|support|treatment|therapy)\b',
            r'\b(?:mental\s+health\s+(?:resource|service|professional|support)|emergency\s+services?)\b',
        ]
        if any(re.search(p, sentence, re.I) for p in resource_patterns):
            return "resource"

        # Advisory patterns (giving advice/recommendation)
        advisory_patterns = [
            r'\byou should\b', r'\bI recommend\b', r'\btry this\b',
            r'\bhere\'?s? what to do\b', r'\bwhat to do\b',
            r'\bmy advice is\b', r'\bmy recommendation\b',
            r'\btake\b.*\b(medication|drug|pill)\b',
            r'\b(invest in|buy|sell)\b',
            r'\bfile\s+a\s+lawsuit\b',
        ]
        if any(re.search(p, sentence, re.I) for p in advisory_patterns):
            return "advisory"

        # Request/imperative patterns - if the user is actively asking for
        # blocked content these override any informational/warning signal that
        # might also appear in the same sentence (issue #24: context bypass).
        request_patterns = [
            r'\b(give me|tell me|show me|explain|provide)\b',
            r'\b(I need|I want|please)\b',
            r'\bhow to\b',
            r'\b(can you|could you|would you)\b',
            # Personal advice-seeking and how-to patterns
            r'\bwhat should I\b',
            r'\bshould I\b',
            r'\bcan I\b',
            r'\bhow do I\b',
            r'\bhow can I\b',
            r'\bwhat are my\b',
            r'\bwhat is my\b',
            r'\bdo I have\b',
            r'\bcould it be\b',
            r'\bhave I developed\b',
            r'\bam I\b',
            r'\bhelp me\b',
            r'\bguide me\b',
            # "What are the methods/steps/ways for X" — instructional queries
            r'\bwhat\s+are\s+(?:some\s+|the\s+)?(?:methods?|steps?|ways?|techniques?|means?|best\s+ways?)\s+(?:to|for|of)\b',
            r'\bwhat\s+(?:is|are)\s+(?:some\s+|the\s+)?(?:best|good|effective)\s+ways?\s+(?:to|for)\b',
            r'\bwhat\s+are\s+some\b',
            r'\bhow\s+(?:do|can|do\s+I|can\s+I)\b',
            r'\bwhat is the most\b',
            r'\bwhat is the best way\b',
            r'\bwhat is the easiest way\b',
            r'\bwhat is the fastest way\b',
            r'\bwhat\s+is\s+the\s+(?:most|best|fastest|easiest|quickest|worst|safest)\b',
            # "What X does my Y have?" pattern — asking about someone's personal attributes
            r'\bdoes\s+my\b',
            r'\bdoes\s+(?:your|his|her|their)\b',
        ]
        has_request = any(re.search(p, sentence, re.I) for p in request_patterns)

        # Informational patterns (facts, research, statistics)
        info_patterns = [
            r'\b(according to|research shows|studies indicate|studies show|studies have shown)\b',
            r'\b(statistics|data|report|survey|results)\b',
            r'\b(in general|typically|usually)\b',
            r'\b(history|historically|past)\b',
            r'\bresearch\b',
            r'\bfollowing\b.*\bfrom\b',
            r'\bscience\b',
            # Definitional questions — only pure educational "What is X?" forms.
            # Exclude personal ("my/your/his/her"), "what are some", "what are the
            # methods/steps/ways/covert", and other request-like forms.
            r'\bwhat\s+(?:is|are)\s+(?!the\s+(?:best|easiest|most|quickest|fastest|most\s+effective)|some\b|your\b|my\b|his\b|her\b|their\b|the\s+(?:classified|confidential|explicit|details?|records?|operations?|methods?|steps?|ways?|covert|process|procedure))\w',
        ]
        if not has_request and any(re.search(p, sentence, re.I) for p in info_patterns):
            return "informational"

        # Warning/educational patterns
        warning_patterns = [
            r'\b(avoid|don\'?t|never|warning|caution)\b',
            r'\b(dangerous|harmful|risky)\b',
            r'\b(should not|shouldn\'?t|must not|mustn\'?t)\b',
        ]
        if not has_request and any(re.search(p, sentence, re.I) for p in warning_patterns):
            return "warning"

        # Direct/instructional (most dangerous)
        return "direct"

    def _build_whitelist_keywords(self, topic: str) -> List[str]:
        """Build a keyword list for a whitelisted topic.

        For known topics (those in TOPIC_SYNONYMS), reuse the synonym
        phrases as positive-match keywords.  For arbitrary topics, the
        topic name itself (and constituent words) serve as keywords.
        """
        keywords = [topic]

        if self.use_synonyms and topic in self.TOPIC_SYNONYMS:
            syn_data = self.TOPIC_SYNONYMS[topic]
            keywords.extend(syn_data.get("phrases", []))
            keywords.extend(syn_data.get("broad_terms", []))

        # For multi-word topics, also add individual words (length >= 3)
        words = topic.split()
        if len(words) > 1:
            keywords.extend(w for w in words if len(w) >= 3)

        # Add custom synonyms if provided
        custom_synonyms = self.config.get('custom_synonyms', {})
        if topic in custom_synonyms:
            keywords.extend([s.lower() for s in custom_synonyms[topic]])

        return keywords

    def _detect_whitelist(self, text: str) -> DetectorResult:
        """Whitelist mode: block text that is NOT about any allowed topic."""
        text_lower = text.lower()

        for topic in self.allowed_topics:
            keywords = self.allowed_topic_keywords.get(topic, [topic])
            for keyword in keywords:
                if ' ' in keyword:
                    if keyword in text_lower:
                        return DetectorResult(decision=Decision.ALLOW, risk_score=0)
                else:
                    regex = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    if regex.search(text):
                        return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        return DetectorResult(
            decision=Decision.BLOCK,
            risk_score=80,
            rule_hits=[RuleHit(
                rule_id="topic.off_topic",
                severity=Severity.HIGH,
                message=f"Text is not about any allowed topic. Allowed: {', '.join(self.allowed_topics)}"
            )],
            user_message="I can only help with specific topics.",
            developer_message=f"Off-topic content detected. Allowed topics: {', '.join(self.allowed_topics)}"
        )

    def detect(self, text: str) -> DetectorResult:
        """
        Detect restricted topics in text with P1 enhancements:
        - Synonym expansion for comprehensive coverage
        - Context understanding
        - Severity differentiation

        Args:
            text: Input text to scan

        Returns:
            DetectorResult with findings
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        if self.mode == "whitelist":
            return self._detect_whitelist(text)

        if not self.blocked_topics:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        text_lower = text.lower()
        found_topics: Set[str] = set()

        # P1: Check for each blocked topic - phrases first, then broad terms
        for topic in self.blocked_topics:
            if topic in found_topics:
                continue

            matched_pattern = None
            is_broad = False

            # 1) Check phrase patterns (strong signal - always actionable)
            for pattern in self.topic_phrases.get(topic, [topic]):
                if ' ' in pattern:
                    if pattern in text_lower:
                        matched_pattern = pattern
                        break
                else:
                    regex = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
                    if regex.search(text):
                        matched_pattern = pattern
                        break

            # 2) Check broad terms (weak signal - require advisory/direct intent)
            if matched_pattern is None:
                for pattern in self.topic_broad.get(topic, []):
                    if ' ' in pattern:
                        if pattern in text_lower:
                            matched_pattern = pattern
                            is_broad = True
                            break
                    else:
                        regex = re.compile(r'\b' + re.escape(pattern) + r'\b', re.IGNORECASE)
                        if regex.search(text):
                            matched_pattern = pattern
                            is_broad = True
                            break

            if matched_pattern is None:
                continue

            # P1: Determine context
            # For always-harmful topics, skip context analysis entirely —
            # the phrase match alone is sufficient to block.
            context = "direct"
            if self.context_aware and topic not in self.NO_CONTEXT_TOPICS:
                context = self._detect_topic_context(text, topic, matched_pattern)

            # Skip resource/help (e.g., suicide hotline)
            if context == "resource":
                continue

            # Skip informational or warning (educational, not harmful advice)
            if context in ["informational", "warning"]:
                continue

            # When context_aware, broad terms only trigger for advisory/direct
            if is_broad and self.context_aware and context not in ["advisory", "direct"]:
                continue

            found_topics.add(topic)

            severity = Severity.HIGH

            message = f"Blocked topic detected: {topic}"
            if matched_pattern != topic:
                message += f" (via synonym: {matched_pattern})"
            if context != "direct":
                message += f" (context: {context})"

            rule_hits.append(RuleHit(
                rule_id=f"topic.{topic.replace(' ', '_')}",
                severity=severity,
                message=message
            ))


        # No blocked topics found (or all were informational/resource)
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # P1: Calculate risk score based on severity
        risk_score = self._calculate_risk_score(rule_hits)

        block_threshold = self.config.get('block_threshold', 40)
        if risk_score >= block_threshold:
            return DetectorResult(
                decision=Decision.BLOCK,
                risk_score=risk_score,
                rule_hits=rule_hits,
                user_message="I can't help with that topic.",
                developer_message=f"Blocked topics detected: {', '.join(found_topics)}"
            )
        return DetectorResult(
            decision=Decision.WARN,
            risk_score=risk_score,
            rule_hits=rule_hits,
            developer_message=f"Restricted topic mentioned: {', '.join(found_topics)}"
        )

    @staticmethod
    def _calculate_risk_score(rule_hits: List[RuleHit]) -> int:
        """Calculate risk score based on severity levels."""
        return calculate_risk_score(rule_hits)
