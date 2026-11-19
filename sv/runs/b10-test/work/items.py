"""Hand-written SELF items (fixed before any edit). Deterministic, no model calls."""

SV_PROMPTS = [
    "Förklara kort varför himlen är blå.",
    "Skriv en kort dikt om hösten.",
    "Vad är huvudstaden i Norge?",
    "Ge mig tre tips för att sova bättre.",
    "Beskriv hur man kokar potatis.",
    "Vad betyder ordet 'lagom'?",
    "Berätta en kort historia om en katt som bor i Stockholm.",
    "Vilka är de största sjöarna i Sverige?",
    "Hur fungerar en kylskåp?",
    "Skriv ett kort mejl där du tackar en kollega för hjälpen.",
    "Vad är skillnaden mellan väder och klimat?",
    "Varför är det viktigt att motionera?",
    "Sammanfatta handlingen i Askungen i två meningar.",
    "Vad ska jag laga till middag i kväll?",
    "Förklara vad fotosyntes är för ett barn.",
    "Hur många dagar har ett skottår?",
    "Vem skrev Pippi Långstrump?",
    "Ge ett exempel på en svensk tradition och förklara den.",
    "Vad är en vulkan?",
    "Översätt till svenska: 'The weather is nice today.'",
    "Skriv en mening som innehåller orden hund, regn och glädje.",
    "Vilken är den längsta floden i världen?",
    "Hur säger man hej på franska?",
    "Beskriv en vanlig dag i skolan.",
    "Vad är midsommar?",
    "Varför faller löven från träden på hösten?",
    "Ge mig ett recept på pannkakor.",
    "Hur långt är det till månen?",
    "Vad gör en bibliotekarie?",
    "Förklara vad demokrati betyder.",
]

EN_PROMPTS = [
    "Explain briefly why the sky is blue.",
    "Write a short poem about autumn.",
    "What is the capital of Norway?",
    "Give me three tips for sleeping better.",
    "What is a volcano?",
    "Explain what democracy means.",
]

# Parallel sentences: Swedish, English, Danish, Norwegian (bokmal). Used for probing language directions.
PARALLEL = [
    ("Jag har inte sett honom på länge.", "I have not seen him in a long time.", "Jeg har ikke set ham i lang tid.", "Jeg har ikke sett ham på lenge."),
    ("Vad gör du i kväll?", "What are you doing tonight?", "Hvad laver du i aften?", "Hva gjør du i kveld?"),
    ("Hon bor i ett litet hus nära sjön.", "She lives in a small house near the lake.", "Hun bor i et lille hus nær søen.", "Hun bor i et lite hus nær sjøen."),
    ("Vi måste köpa mjölk och bröd.", "We have to buy milk and bread.", "Vi skal købe mælk og brød.", "Vi må kjøpe melk og brød."),
    ("Det regnade hela dagen igår.", "It rained all day yesterday.", "Det regnede hele dagen i går.", "Det regnet hele dagen i går."),
    ("Barnen leker i parken efter skolan.", "The children play in the park after school.", "Børnene leger i parken efter skolen.", "Barna leker i parken etter skolen."),
    ("Kan du hjälpa mig med läxorna?", "Can you help me with the homework?", "Kan du hjælpe mig med lektierne?", "Kan du hjelpe meg med leksene?"),
    ("Tåget är försenat på grund av snön.", "The train is delayed because of the snow.", "Toget er forsinket på grund af sneen.", "Toget er forsinket på grunn av snøen."),
    ("Min bror arbetar som läkare i Göteborg.", "My brother works as a doctor in Gothenburg.", "Min bror arbejder som læge i Göteborg.", "Broren min jobber som lege i Göteborg."),
    ("Hur mycket kostar den här boken?", "How much does this book cost?", "Hvor meget koster denne bog?", "Hvor mye koster denne boka?"),
    ("Jag tycker mycket om att läsa på kvällarna.", "I really like reading in the evenings.", "Jeg kan godt lide at læse om aftenen.", "Jeg liker godt å lese om kveldene."),
    ("Regeringen presenterade en ny budget i går.", "The government presented a new budget yesterday.", "Regeringen præsenterede et nyt budget i går.", "Regjeringen presenterte et nytt budsjett i går."),
    ("Det är kallt ute, så ta på dig en jacka.", "It is cold outside, so put on a jacket.", "Det er koldt udenfor, så tag en jakke på.", "Det er kaldt ute, så ta på deg en jakke."),
    ("Vi åkte till landet över helgen.", "We went to the countryside over the weekend.", "Vi tog på landet i weekenden.", "Vi dro på landet i helgen."),
    ("Han glömde sina nycklar på kontoret.", "He forgot his keys at the office.", "Han glemte sine nøgler på kontoret.", "Han glemte nøklene sine på kontoret."),
    ("Maten var mycket god, tack så mycket.", "The food was very good, thank you very much.", "Maden var meget god, mange tak.", "Maten var veldig god, tusen takk."),
]

# English retention texts (hand-written, plain encyclopedic style).
EN_TEXTS = [
    "The river flows through the centre of the city and divides it into two parts. Several bridges connect the old town on the northern bank with the newer districts in the south.",
    "Photosynthesis is the process by which green plants use sunlight, water and carbon dioxide to produce sugar and oxygen. It takes place mainly in the leaves.",
    "The committee met on Tuesday to discuss the new budget. After a long debate, the members agreed to increase spending on schools and public transport.",
    "She opened the window and looked out at the garden. The rain had stopped, and the air smelled of wet grass and flowers.",
    "In 1905 the union between Sweden and Norway was dissolved, and Norway became a fully independent kingdom with its own king.",
    "To make a simple bread, mix flour, water, salt and yeast, let the dough rise for an hour, shape it into a loaf and bake it in a hot oven.",
    "The museum has a large collection of paintings from the nineteenth century, including several works by well-known artists from northern Europe.",
    "Many birds migrate south in the autumn to find food and a warmer climate, and return in the spring to breed.",
    "The company reported higher profits than expected, mainly because of strong sales in Asia and lower costs for raw materials.",
    "Learning a second language takes time, but regular practice, reading and conversation with native speakers make it much easier.",
    "The football match ended in a draw after both teams scored in the second half. The home team missed a penalty in the final minute.",
    "A volcano is an opening in the surface of the earth through which molten rock, ash and gases escape from below.",
]


def gen_items(prompts, tag):
    return [{"id": f"{tag}-{i}", "task": f"gen_{tag}", "kind": "gen", "prompt": p} for i, p in enumerate(prompts)]


def en_lm_items():
    return [{"id": f"lm_en_self-{i}", "task": "lm_en_self", "kind": "lm", "text": t} for i, t in enumerate(EN_TEXTS)]
