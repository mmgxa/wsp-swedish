"""Hand-written parallel paragraphs (Swedish / Norwegian Bokmal / Danish / English) for language-neuron probing.
Written by me for this run; not taken from any evaluation split."""

PAR = [
    {
        "sv": "Jag vet inte om hon kommer i dag. Hon sa att hon måste arbeta sent, men hon skulle försöka ringa mig när hon var klar.",
        "no": "Jeg vet ikke om hun kommer i dag. Hun sa at hun måtte jobbe sent, men hun skulle prøve å ringe meg når hun var ferdig.",
        "da": "Jeg ved ikke, om hun kommer i dag. Hun sagde, at hun skulle arbejde sent, men hun ville prøve at ringe til mig, når hun var færdig.",
        "en": "I do not know if she is coming today. She said that she had to work late, but she would try to call me when she was done.",
    },
    {
        "sv": "Det är svårt att säga exakt hur många människor som bodde i staden på den tiden. De flesta källor är osäkra och många uppgifter saknas.",
        "no": "Det er vanskelig å si nøyaktig hvor mange mennesker som bodde i byen på den tiden. De fleste kilder er usikre, og mange opplysninger mangler.",
        "da": "Det er svært at sige præcist, hvor mange mennesker der boede i byen på den tid. De fleste kilder er usikre, og mange oplysninger mangler.",
        "en": "It is hard to say exactly how many people lived in the city at that time. Most sources are uncertain and much information is missing.",
    },
    {
        "sv": "Vi åkte till landet i somras och bodde i en liten stuga vid sjön. Varje morgon badade barnen, och på kvällarna grillade vi och spelade kort.",
        "no": "Vi dro på landet i sommer og bodde i en liten hytte ved sjøen. Hver morgen badet barna, og om kveldene grillet vi og spilte kort.",
        "da": "Vi tog på landet i sommer og boede i et lille sommerhus ved søen. Hver morgen badede børnene, og om aftenen grillede vi og spillede kort.",
        "en": "We went to the countryside last summer and stayed in a small cabin by the lake. Every morning the children swam, and in the evenings we had barbecues and played cards.",
    },
    {
        "sv": "Regeringen har beslutat att höja skatten på bensin från och med nästa år. Kritiker menar att beslutet drabbar dem som bor på landsbygden hårdast.",
        "no": "Regjeringen har besluttet å øke avgiften på bensin fra og med neste år. Kritikere mener at vedtaket rammer dem som bor på bygda hardest.",
        "da": "Regeringen har besluttet at hæve afgiften på benzin fra og med næste år. Kritikere mener, at beslutningen rammer dem, der bor på landet, hårdest.",
        "en": "The government has decided to raise the tax on petrol from next year. Critics argue that the decision hits those who live in rural areas the hardest.",
    },
    {
        "sv": "Hunden sprang ut i trädgården så fort dörren öppnades. Den jagade en ekorre ända bort till staketet och skällde högt när ekorren klättrade upp i ett träd.",
        "no": "Hunden løp ut i hagen så snart døren ble åpnet. Den jaget et ekorn helt bort til gjerdet og bjeffet høyt da ekornet klatret opp i et tre.",
        "da": "Hunden løb ud i haven, så snart døren blev åbnet. Den jagede et egern helt hen til hegnet og gøede højt, da egernet klatrede op i et træ.",
        "en": "The dog ran out into the garden as soon as the door was opened. It chased a squirrel all the way to the fence and barked loudly when the squirrel climbed up a tree.",
    },
    {
        "sv": "Kan du hjälpa mig med en sak? Jag behöver låna din bil i morgon eftersom min egen är på verkstaden. Jag lovar att köra försiktigt.",
        "no": "Kan du hjelpe meg med en ting? Jeg trenger å låne bilen din i morgen fordi min egen er på verkstedet. Jeg lover å kjøre forsiktig.",
        "da": "Kan du hjælpe mig med noget? Jeg har brug for at låne din bil i morgen, fordi min egen er på værkstedet. Jeg lover at køre forsigtigt.",
        "en": "Can you help me with something? I need to borrow your car tomorrow because my own is at the garage. I promise to drive carefully.",
    },
    {
        "sv": "Fotosyntesen är den process där växter använder solljus för att omvandla koldioxid och vatten till socker och syre. Utan den skulle livet på jorden inte vara möjligt.",
        "no": "Fotosyntesen er prosessen der planter bruker sollys til å omdanne karbondioksid og vann til sukker og oksygen. Uten den ville livet på jorden ikke vært mulig.",
        "da": "Fotosyntesen er den proces, hvor planter bruger sollys til at omdanne kuldioxid og vand til sukker og ilt. Uden den ville livet på jorden ikke være muligt.",
        "en": "Photosynthesis is the process in which plants use sunlight to convert carbon dioxide and water into sugar and oxygen. Without it, life on earth would not be possible.",
    },
    {
        "sv": "Laget vann matchen med tre mål mot ett. Tränaren var mycket nöjd efteråt och sa att spelarna hade kämpat hårt under hela andra halvleken.",
        "no": "Laget vant kampen med tre mål mot ett. Treneren var svært fornøyd etterpå og sa at spillerne hadde kjempet hardt gjennom hele andre omgang.",
        "da": "Holdet vandt kampen med tre mål mod et. Træneren var meget tilfreds bagefter og sagde, at spillerne havde kæmpet hårdt gennem hele anden halvleg.",
        "en": "The team won the match three goals to one. The coach was very pleased afterwards and said that the players had fought hard throughout the whole second half.",
    },
    {
        "sv": "Min mormor föddes i en liten by i norra Sverige. Hon berättade ofta om de kalla vintrarna, när snön låg meterhög och skolan ibland var stängd i flera veckor.",
        "no": "Min mormor ble født i en liten bygd i Nord-Norge. Hun fortalte ofte om de kalde vintrene, da snøen lå meterhøy og skolen noen ganger var stengt i flere uker.",
        "da": "Min mormor blev født i en lille landsby i Nordjylland. Hun fortalte ofte om de kolde vintre, hvor sneen lå meterhøj, og skolen nogle gange var lukket i flere uger.",
        "en": "My grandmother was born in a small village in the north. She often talked about the cold winters, when the snow lay a metre deep and the school was sometimes closed for weeks.",
    },
    {
        "sv": "Vad tycker du om den nya filmen? Jag tyckte att den var lite för lång, men skådespelarna var mycket bra och musiken var fantastisk.",
        "no": "Hva synes du om den nye filmen? Jeg syntes den var litt for lang, men skuespillerne var veldig gode og musikken var fantastisk.",
        "da": "Hvad synes du om den nye film? Jeg syntes, den var lidt for lang, men skuespillerne var meget gode, og musikken var fantastisk.",
        "en": "What do you think of the new film? I thought it was a bit too long, but the actors were very good and the music was fantastic.",
    },
    {
        "sv": "Företaget meddelade på tisdagen att man kommer att säga upp omkring tvåhundra anställda. Fackförbundet kallar beskedet för ett hårt slag mot hela regionen.",
        "no": "Selskapet meldte tirsdag at det vil si opp rundt to hundre ansatte. Fagforeningen kaller beskjeden et hardt slag mot hele regionen.",
        "da": "Virksomheden meddelte tirsdag, at den vil afskedige omkring to hundrede ansatte. Fagforbundet kalder meldingen et hårdt slag mod hele regionen.",
        "en": "The company announced on Tuesday that it will lay off around two hundred employees. The union calls the news a hard blow to the whole region.",
    },
    {
        "sv": "Det finns många sätt att lära sig ett nytt språk. Det viktigaste är att man övar varje dag, läser böcker och inte är rädd för att göra misstag när man pratar.",
        "no": "Det finnes mange måter å lære seg et nytt språk på. Det viktigste er at man øver hver dag, leser bøker og ikke er redd for å gjøre feil når man snakker.",
        "da": "Der er mange måder at lære et nyt sprog på. Det vigtigste er, at man øver sig hver dag, læser bøger og ikke er bange for at lave fejl, når man taler.",
        "en": "There are many ways to learn a new language. The most important thing is to practise every day, read books and not be afraid of making mistakes when you speak.",
    },
    {
        "sv": "Efter kriget flyttade familjen till Göteborg, där fadern fick arbete på ett varv. Barnen gick i skolan i stadsdelen och lärde sig snabbt det nya språket.",
        "no": "Etter krigen flyttet familien til Bergen, der faren fikk arbeid på et verft. Barna gikk på skole i bydelen og lærte seg raskt det nye språket.",
        "da": "Efter krigen flyttede familien til Aarhus, hvor faderen fik arbejde på et værft. Børnene gik i skole i bydelen og lærte hurtigt det nye sprog.",
        "en": "After the war the family moved to the coast, where the father found work at a shipyard. The children went to school in the district and quickly learned the new language.",
    },
    {
        "sv": "Hur lång tid tar det att åka tåg från Stockholm till Malmö? Det tar ungefär fyra och en halv timme, men ibland blir det förseningar på grund av arbeten på spåret.",
        "no": "Hvor lang tid tar det å ta toget fra Oslo til Trondheim? Det tar omtrent seks og en halv time, men noen ganger blir det forsinkelser på grunn av arbeid på sporet.",
        "da": "Hvor lang tid tager det at tage toget fra København til Aalborg? Det tager omkring fire og en halv time, men nogle gange bliver der forsinkelser på grund af sporarbejde.",
        "en": "How long does it take to go by train from the capital to the south? It takes about four and a half hours, but sometimes there are delays because of work on the track.",
    },
    {
        "sv": "Jag är inte säker på att jag kan svara på den frågan. Det beror på vad du menar, och det finns flera olika sätt att se på saken.",
        "no": "Jeg er ikke sikker på at jeg kan svare på det spørsmålet. Det kommer an på hva du mener, og det finnes flere forskjellige måter å se saken på.",
        "da": "Jeg er ikke sikker på, at jeg kan svare på det spørgsmål. Det afhænger af, hvad du mener, og der er flere forskellige måder at se sagen på.",
        "en": "I am not sure that I can answer that question. It depends on what you mean, and there are several different ways to look at the matter.",
    },
    {
        "sv": "Boken handlar om en ung kvinna som lämnar sin hemstad för att studera i huvudstaden. Där möter hon nya vänner, men hon saknar också sin familj mer än hon trodde.",
        "no": "Boken handler om en ung kvinne som forlater hjembyen sin for å studere i hovedstaden. Der møter hun nye venner, men hun savner også familien sin mer enn hun trodde.",
        "da": "Bogen handler om en ung kvinde, der forlader sin hjemby for at studere i hovedstaden. Der møder hun nye venner, men hun savner også sin familie mere, end hun troede.",
        "en": "The book is about a young woman who leaves her home town to study in the capital. There she meets new friends, but she also misses her family more than she expected.",
    },
]
