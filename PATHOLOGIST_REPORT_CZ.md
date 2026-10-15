# Duktální vs lobulární morfologie z tile embeddin­gů — shrnutí pro patologa

## Otázka

Dokáže model čistě z morfologie H&E dlaždic (tiles) — bez klinických dat —
rozlišit invazivní duktální karcinom (8500) od invazivního lobulárního
karcinomu (8520)?

## Metoda ve zkratce

Každé sklíčko je rozřezáno na malá obrázková pole. Virchow2 (patologický
foundation model) vytvoří embedding každého pole, detektor karcinomu vybere
tumorová pole (pravděpodobnost ≥ 0,95) a lineární projekce natrénovaná na
1 435 trénovacích sklíčkách (1 189 duktálních, 246 lobulárních) přiřadí
každému poli skóre: **záporné = duktální, kladné = lobulární**. Skóre sklíčka
= průměr přes jeho tumorová pole; škála je globální (stejná hodnota = stejný
význam na všech sklíčkách). Výsledky níže jsou z validační sady —
293 označených sklíček, které model nikdy neviděl.

## Hlavní výsledek

Na sklíčkách s dostatkem detekovaného tumoru (≥ 100 polí; 196 z 293):

| Metrika (slide level, Val, n=196) | hodnota |
|---|---|
| AUC (lobulární skóruje nad duktálním) | **0,972** |
| Přesnost na přirozené hranici (skóre > 0) | **94,4 %** |
| Vyvážená přesnost | 95,3 % |

Pozn.: lobulárních je jen 31 sklíček, interval spolehlivosti AUC ≈ ±0,03.

## Struktura chyb — proč je to zajímavé

1. **Třetina sklíček jsou negativní bloky (N), správně prázdné.** Podle
   nomenklatury (T = tumor, N = negativ, M = metastázy) na nich detektor
   karcinom správně nenachází — téměř žádná tumorová pole, medián 4. Tato
   sklíčka ale zdědí morfologii případu a dostala se tak do vyhodnocení,
   kde jejich průměrné skóre byl šum. Po jejich vynechání se AUC zvedla z
   0,878 na 0,972.
2. **Dva dobře obsazené sklíčka odporují označení** — skutečné rozpory:
   `2020_08267-04-T2` (označeno 8520, 27 tis. polí, skóre −1,48 = jednoznačně
   duktální) a `2023_08660-2-T` (označeno 8500, 46 tis. polí, skóre +1,63 =
   jednoznačně lobulární). Hypotéza: smíšený tumor nebo nesoulad v kódování
   morfologie.
3. Zbylých ~45 chyb leží u hranice (|skóre| < 0,5) — skutečně nejednoznačné
   případy, jak to u překrývajících se rozdělení čekáme.

## Co vás žádíme

V xOpat reportu je skóre modelu vykreslené přes WSI: **modrá = duktální
políčka, červená = lobulární**, nejistý střední pás skrytý. Prosíme:

1. **Kalibrace** — odpovídají modré oblasti duktálnímu vzoru a červené
   lobulárnímu (řetízkový růst, diskohese)?
2. **Rozpory** — u dvou sklíček výše: jde o smíšený tumor, chybné označení,
   nebo chybu modelu?
3. **Prázdná sklíčka** — pokud na některém N-bloku maska ukazuje zásadní
   barevné oblasti, jde o false positive detektoru karcinomu.

## Bonus

Vzácnější morfologie (netrénované) skórují souvisle na stejné ose: 8522
(lobulární + DCIS) a 8509 (papilární) se kloní k lobulárnímu; 8503
(tubulární) a 8504 (kribiformní) k duktálnímu. Osa zřejmě zachycuje biologii
růstového vzoru, ne jen dvě trénované třídy.

## Výsledek posouzení patologem

Oba skutečné rozpory byly **chyby v označení, ne chyby modelu**:

- `2020_08267-04-T2` — pacientka má více nádorů v obou prsech; tento blok
  je z karcinomu pravého prsu a je skutečně duktální (8500). V tabulce byl
  chybně přiřazen k (lobulárnímu) tumoru levého prsu; blok T1 je ten s
  8520. Model měl pravdu (skóre −1,48).
- `2023_08660-2-T` — skutečně lobulární karcinom; chyba v kódování, jejíž
  korekce se nedostala do exportu z databáze. Model měl pravdu (skóre +1,63).

Po vyřešení obou případů model nemá na dobře obsazených sklíčkách žádné
nevysvětlené rozpory — všechny zbývající chyby leží u rozhodovací hranice.

**Otevřená otázka:** u sklíček se smíšenými labely se patologovi nepodařilo
vizuálně definovat rozdíl mezi modře a červeně zbarvenými poli. Osa zřetelně
odděluje labely na úrovni sklíčka (AUC 0,972), ale její políčkové důkazy
žijí v 2560rozměrných rysech foundation modelu, které nemusí odpovídat
architektonickým vzorům, které lidé pojmenovávají — signál může být jemná
textura či statistika, nikoli řetízkový vs glandulární růst. Přirozeným
dalším krokem je prozkoumat, zda jsou vizuálně odlišná pole s extrémním
skóre u smíšených tumorů.

*Detaily k dispozici: Virchow2 embeddings, filtr ≥ 0,95, LDA se shrinkage,
sady Train/Val, výstupy v `/mnt/projects/breast_cancer/tile_morph_clustering/`.*
