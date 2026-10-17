# Duktální vs lobulární morfologie z tile embeddin­gů — shrnutí pro patologa

*v6, říjen 2025: všechna skóre přepočítána po opravě zarovnání souřadnic
polí ve scoring pipeline (skóre byla zapisována na špatné souřadnice při
filtrování na karcinomová pole). Posouzení patologem níže proběhlo na v5
maskách; obě jeho závěry byly pod v6 ověřeny a platí — s výraznějšími skóre.*

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
| AUC (lobulární skóruje nad duktálním) | **0,976** |
| Přesnost na přirozené hranici (skóre > 0) | **98,0 %** |
| Vyvážená přesnost | 96,2 % |

Pozn.: lobulárních je jen 31 sklíček, interval spolehlivosti AUC ≈ ±0,03.

## Struktura chyb — proč je to zajímavé

97 z 293 validačních sklíček nemá téměř žádný detekovaný tumor (medián 4
polí) — většinou negativní bloky (N), které zdědí morfologii případu přes
join podle record-prefixu. Jejich průměrné skóre je šum, proto jsou
vynechána (< 100 polí); čísla výše platí pro zbývajících 196 sklíček.

Z těchto 196 jich na špatné straně hranice skončí jen **4**:

1. **Dvě jsou známé chyby v označení** (viz níže): `2020_08267-04-T2`
   (označeno 8520, 27 tis. polí, skóre −3,28 = jednoznačně duktální) a
   `2023_08660-2-T` (označeno 8500, 46 tis. polí, skóre +3,90 = jednoznačně
   lobulární). Patolog obě potvrdil jako chyby tabulky/kódování; model měl
   v obou případech pravdu.
2. **Dvě leží na hranici**: `2021_05355-02-T` (označeno 8520, skóre −0,20)
   a `2024_00961-2-N` (označeno 8500, skóre +0,05 — N blok, který zdědil
   label případu, 580 polí). Skutečně nejednoznačné případy.

Na dobře obsazených sklíčkách tedy model **nemá žádnou nevysvětlenou chybu**
— každá špatná klasifikace je buď doložená chyba labelu, nebo hod mincí
na hranici.

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
  8520. Model měl pravdu (v6 skóre −3,28).
- `2023_08660-2-T` — skutečně lobulární karcinom; chyba v kódování, jejíž
  korekce se nedostala do exportu z databáze. Model měl pravdu (v6 skóre +3,90).

Po vyřešení obou případů model nemá na dobře obsazených sklíčkách žádné
nevysvětlené rozpory — všechny zbývající chyby leží u rozhodovací hranice.

**Otevřená otázka:** u sklíček se smíšenými labely se patologovi nepodařilo
vizuálně definovat rozdíl mezi modře a červeně zbarvenými poli. Pozor:
toto posouzení proběhlo na v5 maskách, které měly bug zarovnání souřadnic
— modré/červené hodnoty seděly na špatných souřadnicích, což samo může
vysvětlovat, proč nebyl nalezen žádný vizuální vzor. V6 masky kladou každou
hodnotu na její skutečné pole; zaslouží si čerstvý pohled, zda jsou pole s
extrémním skóre vizuálně odlišná. Osa zřetelně odděluje labely na úrovni
sklíčka (AUC 0,976), ale její políčkové důkazy žijí v 2560rozměrných rysech
foundation modelu, které nemusí odpovídat architektonickým vzorům, které
lidé pojmenávají — signál může být jemná textura či statistika, nikoli
řetízkový vs glandulární růst.

*Detaily k dispozici: Virchow2 embeddings, filtr ≥ 0,95, LDA se shrinkage,
sady Train/Val, výstupy v `/mnt/projects/breast_cancer/tile_morph_clustering/`.*
