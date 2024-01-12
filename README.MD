Jos löydät virheitä, haluat lisätä jotain tai on jotain kysyttävää voit tehdä pull requestin, tehdä issuen tai lähettää sähköpostia osoitteeseen:

> saloranta.jonne[at]gmail.com

## Vaatimukset

<details>
<summary>Vaadittavat ohjelmat</summary>

<br>

[Colab (vaatii Google käyttäjän)](https://colab.research.google.com/)
Colabia Käytetään mallien opettamiseen ja testaamiseen, jos sinulla ei ole omaa konetta jossa on tarpeeksi tehoa ja CUDA yhteensopiva näytönohjain.

Tässä ohjeessa käytetään Colabia, joten tämä ei sisällä ohjeita sille miten CUDA yms asennetaan/käytetään.

[Python](https://www.python.org/downloads/) Python >=3.8, koska Ultralytics vaatii sen.

[Ultralytics](https://github.com/ultralytics/ultralytics) Ultralytics on paketti joka sisältää paljon erilaisia koneoppimis malleja. Tässä ohjeessa käytetään YOLOv8 mallia.

[Ultralytics HUB (vaatii kirjautumisen)](https://hub.ultralytics.com/) Täältä löytyy ilmaisia malleja, joita voi käyttää.

[Ultralytics (ENG.)](https://docs.ultralytics.com/) Täältä löytyy dokumentointia ja ohjeita Ultralyticsin käyttöön.

[LabelImg](https://github.com/HumanSignal/labelImg) on ohjelma jolla voi luoda kuvista annotaatioita. Annotaatiot ovat tiedostoja, jotka sisältävät tiedot kuvan tunnistettavista objekteista.

[Ohjevideo Colabiin (YouTube)](https://youtu.be/WgPbbWmnXJ8)

</details>

## Sisällysluettelo

<details>
<summary>Sisällysluettelo</summary>

<br>

### Vaatimukset

-   [x] [Mallipohjan lataaminen](#mallipohjan-lataaminen)
-   [x] [Virtuaalinen ympäristö](#virtuaalinen-ympäristö)

### Kuvat ja niiden käsittely

-   [x] [Kuvien ottaminen](#kuvien-ottaminen)
-   [x] [Kuvien käsittely](#kuvien-käsittely)
-   [x] [Kuvien muokkaus scripti](#randomize_images.py)

</details>

<details>
<summary>Pohjan lataaminen</summary>

<br>

<a name="mallipohjan-lataaminen"></a>

## Mallipohjan lataaminen

Lataa mallipohja (suositeltavaa) [täältä](https://github.com/JonneSaloranta?tab=repositories) ja pura se haluamaasi kansioon. Voit myös kloonata sen gitillä.

```cmd
git clone https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet.git
```

</details>

<a name="virtuaalinen-ympäristö"></a>

## Virtuaalinen ympäristö

<details>
<summary>Virtuaalisen ympäristön ohje</summary>
<br>

Virtuaalinen ympäristö kannattaa luoda, jotta ei tule ongelmia eri versioiden kanssa. Tässä ohjeessa käytetään Pythonin omaa virtuaalista ympäristöä.

Avaa komentokehote ja siirry kansioon johon haluat luoda virtuaalisen ympäristön.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/4263bfae-9e71-4b83-bdff-fe74918119f0)

```cmd
cd C:\Users\user\Desktop\malli
```

Varmista että Python on asennettu ja lisätty ympäristömuuttujiin.

```cmd
python --version
```

jos sinulla on python asennettu oikein, voit jatkaa ja luoda virtuaalisen ympäristö komennolla:

```cmd
python -m venv [kansion nimi]

esim. venv = virtual environment
```

```cmd
python -m venv venv
```

Nyt voit asentaa Ultralyticsin virtuaaliseen ympäristöön.

```cmd
pip install ultralytics
```

</details>

<a name="kuvien-ottaminen"></a>

## Kuvat ja niiden käsittely eri variaatioilla

<details>
<summary>Kuvien ottamisen ohje</summary>

<br>

Kappaleen tunnistus mallin opetusdatan(kuvien) tulisi sisältää mahdollisimman paljon erilaisia variaatiota esim. valaistus, kuvakulma jne. Opetusdatan kuvien määrä on myös tärkeä tekijä. Mitä enemmän kuvia sitä parempi(useimmiten).

### Kuvien ottaminen

-   [x] Erilaiset kuvakulmat
-   [x] Erilaiset valaistukset
-   [x] Erilaiset taustat
-   [x] Erilaiset kameran zoomaukset
-   [x] Erilaiset kameran tarkennukset

</details>

<a name="kuvien-käsittely"></a>

### Kuvien käsittely

<details>
<summary>Kuvien käsittelyn ohje</summary>

<br>

Paras tapa saada mahdollisimman paljon erilaisia realistisia variaatioita kuvista on ottaa niitä itse, tällöin niistä on mahdollisesti helpompi saada hyvää dataa. Suosittelen jättämään seuraavan kuvien käsittely osion väliin, mutta jos haluat käyttää sitä, niin seuraavassa on ohjeet.

Kuvien käsittely on suhteellisen helppoa. Kuvien käsittelyssä on hyvä käyttää erilaisia kuvankäsittelyohjelmia. Esim. GIMP, Photoshop, Paint.net, jne.

Säästän kuitenkin kaikkien aikaa ja hermoja, joten tein lyhyen scriptin Pythonilla, joka käy läpi kaikki kuvat ja muokkaa niitä satunnaisesti omien raja-arvojen mukaan.

> Yksi syötetty kuva

![Yksi syötetty kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/882321ca-f714-49da-810a-54e99ab78026)

> 50kpl käsiteltyjä kuvia

![50kpl käsiteltyjä kuvia](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/85b98060-6221-4cf9-bb93-c1b14d1ae2d0)

Scripti löytyy täältä: [randomize_images.py](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/blob/master/randomize_images.py)

<a name="randomize_images.py"></a>

```python
num_variations = 5  # Määrittää kuinka monta kuvaa muokattua kuvaa luodaan per alkuperäinen kuva.

# Tämä on varmaan se tärkein parametri. Muitakin voi muokata jos välttämättä haluaa.

# Eri arvoja jotka määrittävät kuinka paljon kuvaa muokataan.

rotation_limit = 180  # Määrittää kuvan maksimi kiertokulman
resize_min = 0.5  # Minimi skaalauskerroin kuvan pienentämiseen
resize_max = 2.0  # Maksimi skaalauskerroin kuvan suurentamiseen
brightness_min = 0.5  # Minimi kirkkauskerroin
brightness_max = 1.5  # Maksimi kirkkauskerroin
contrast_min = -25  # Minimi kontrastikerroin
contrast_max = 25  # Maksiimi kontrastikerroin
saturation_min = 0.5  # Minimi värikylläisyyskerroin
saturation_max = 1.5  # Makismi värikylläisyyskerroin
exposure_min = 0.5  # Minimi valotuskerroin
exposure_max = 1.5  # Maksimi valotuskerroin
```

</details>

## Kuvien annotointi

<details>

<summary>Kuvien annotointi ohje</summary>

<br>

Kuvien annotointi on tärkeä osa mallin opetusdataa. Annotaatiot ovat tiedostoja, jotka sisältävät tiedot kuvan tunnistettavista objekteista ja niiden sijainnista kuvassa.

Annotaatiot voi luoda monella eri ohjelmalla. Tässä ohjeessa käytetään LabelImg ohjelmaa.

[LabelImg Github](https://github.com/HumanSignal/labelImg) ja lataa se windowsille 'releases' kohdasta.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/07375ae6-1a5c-4f1b-999c-46dc9c9d6e5e)

Lataamisen jälkeen pura se haluamaasi kansioon ja avaa kyseisessä kansiossa oleva data kansio ja siellä oleva 'predefined_classes.txt' tiedosto.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/a4d7c399-b311-4719-8e62-1bebdfe1daa5)

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/e334c5a9-1562-47be-92c9-63c6757ed9fd)

Jos et löydä kyseistä tiedostoa, voit luoda sen itse. sijaintiin: data/predefined_classes.txt

Tämä tiedosto sisältää kaikki luokat, joita voi käyttää annotoinnissa. Oman tunnistusmallin luomiseen on hyvä aloittaa tyhjästä ja lisätä luokat joita tarvitset. Luokkia voi lisätä myös myöhemmin.

Tässä on predefinied_classes.txt tiedoston sisältö, josta voi ottaa mallia.

```txt
dog
person
cat
tv
car
meatballs
marinara sauce
tomato soup
chicken noodle soup
french onion soup
chicken breast
ribs
pulled pork
hamburger
cavity
```

Kun olet luonut tiedoston, voit avata LabelImg ohjelman ja valita 'Open Dir' ja valita kansion jossa on kuvat joita haluat annotoida.

Kun olet valinnut kansion, voit valita 'Change Save Dir' ja valita kansion johon annotaatio tiedostot tallennetaan. Tämä kansio on hyvä olla sama kuin kuvien kansio.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/de50015b-c09b-4ba6-ac36-d5f550454cfd)

Asetuksia, joita pitää/kannattaa muuttaa:

Pakolliset:

-   Vaihda 'PascalVOC' 'Yolo' muotoon.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/a56e4450-6a79-4b8c-9677-7ac0f1bd8bd9)

Suosittelen:

-   Vaihda 'Auto Save Mode' päälle, jotta annotaatio tiedostot tallentuvat automaattisesti.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/cc260665-75a7-413a-bfc5-659ceecc4974)

-   Merkitse 'Use Default Label' päälle, jotta voit valita luokan listasta, eikä joka kerta tarvitse kirjoittaa luokkaa itse.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/f6bfc74b-857d-421a-bcba-5efd8581c04b)

Kun olet valinnut kansion, voit aloittaa annotoinnin painamalla W-näppäintä ja piirtämällä laatikon objektin ympärille. Kun olet piirtänyt laatikon, voit valita luokan listasta tai kirjoittaa sen itse.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/f8a9ae25-d6ab-4772-ae4c-32178be0719d)

Pikanäppäimet:

-   w = piirrä laatikko
-   d = seuraava kuva
-   a = edellinen kuva
-   ctrl + s = tallenna annotaatio

Kun olet valinnut luokan, voit tallentaa annotaation painamalla 'Save' nappia. 'Auto Save Mode' ollessa päällä, annotaatio tallentuu automaattisesti, kun vaihdat seuraavaan kuvaan.

Nyt kun olemme tutustuneet LabelImg ohjelmaan, voimme aloittaa oikeasti kuvien annotoinnin, joka on tärkein ja aikaa vievin osa mallin luomisessa.

LabelImg:n luomat annotaatio tiedostot ovat txt tiedostoja, jotka sisältävät tiedot kuvan tunnistettavista objekteista. Tiedostot luodaan automaattisesti kun tallennat annotaation.

classes.txt tiedosto sisältää kaikki luokat, joita käytetään annotaatio tiedostoissa ja niiden järjestys on tärkeä. classes.txt tiedosto on samanlainen kuin predefined_classes.txt tiedosto, mutta se sisältää vain käytetyt luokat.

Tässä on esimerkki annotaatio tiedostosta:

```txt
0 0.494280 0.481580 0.216689 0.154176
```

Tiedosto sisältää 5 arvoa, jotka ovat _luokka, x1, y1, x2, y2_

-   _luokka_ on luokka johon objekti kuuluu.
-   _x1_ on annotaation vasemman yläkulman x koordinaatti.
-   _y1_ on annotaation vasemman yläkulman y koordinaatti.
-   _x2_ on annotaation oikean alakulman x koordinaatti.
-   _y2_ on annotaation oikean alakulman y koordinaatti.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/ff9d9a62-94c9-4f8e-9985-efa38b84cc03)

</details>

## Colab projektin luominen

<details>
<summary>Ohjeet</summary>

1. Avaa Google Drive ja luo uusi kansio, johon tallennat kaikki tiedostot.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/cf4d8127-4152-4302-bf6f-db9144dd2c9c)

2. Avaa kansio ja luo uusi Jupiter Notebook painamalla oikealla hiiren näppäimellä ja valitsemalla 'Lisää' ja 'Google Colaboratory'. Tämä luo uuden Jupiter Notebook tiedoston, jonka pitäisi avautua uuteen välilehteen.
   Voit nimeätä tiedoston haluamallasi nimellä, kunhan se on .ipynb muodossa.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/004a6b70-63e0-42ed-a9c2-8bd43ac069f8)

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/c29ce963-0484-4483-8b21-790c51e2875e)

3. Nyt pitää muokata asetuksia siten, että saadaan GPU(näytönohjain) käyttöön CPU(prosessorin) sijaan. Valitse 'Muokkaa' ja 'Työkirjan asetukset'. Sieltä vaihda CPU -> T4 GPU:ksi.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/cf95a9be-eedb-4742-9722-a4c2c81d2d3c)

Tässä paina 'OK' ja 'Tallenna'.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/aec3d234-e1d6-47d4-8bb5-f7889a1a3cad)

4. Nyt voidaan tarkistaa onko GPU käytössä. Paina '+ Koodi' ja kirjoita seuraava koodi kenttään ja aja se painamalla 'Play' nappia.

```
!nvidia-smi
```

Jos kaikki on mennyt oikein, pitäisi näkyä jotain tällaista.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/471cd2f8-40a0-4ba5-b7da-a77a91ac12cc)

5. Seuraavaksi lisätään seuraavat koodit samalla tavalla kuin äskeinenkin.

Tällä asennetaan Ultralytics paketti.

```
!pip install ultralytics
```

Tällä otetaan Ultralytics paketti käyttöön.

```
from ultralytics import YOLO
```

ja lopuksi tällä opetetaan malli.

```
!yolo task=detect mode=train model=yolov8n.pt data=../content/drive/MyDrive/AI/Dataset/malli/dataset.yaml epochs=200 imgsz=640
```

Tässä sinun tulee muokata data= kohdassa oleva polku oikeaksi [katso Google Driven yhdistäminen projektiin](#google-drive-yhdistäminen) ja epochs= kohdassa oleva luku haluamaksesi. Epochs tarkoittaa kuinka monta kertaa malli käy läpi kaikki kuvat. Mitä enemmän epochseja, sitä parempi mallista tulee, mutta se vie enemmän aikaa.

</details>

<a name="google-drive-yhdistäminen"></a>

## Google Driven yhdistäminen Colabiin

<details>

<summary>Ohjeet</summary>

1. Paina kansio kuvaketta.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/7b4021fa-7e32-4635-ba15-d0c93d1b2fbe)

2. Paina 'Ota käyttöön Drive' nappia.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/20a80be6-b344-4830-9de9-4524e7301745)

3. Anna Colabille lupa käyttää Google Drivea.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/aed430a8-fd60-431e-a4b3-8407c1e397cd)

4. Tämän jälkeen pitäisi tulla 'drive' kansio näkyviin.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/908c39b5-71ad-47ed-89ce-7a1ce87e6603)

Tässä tapauksessa Google Driveen luomani kansio on nimeltään 'yolomalli'. nyt voit käyttää sitä polkuna Colabissa.

5. Nyt meidän tarvitsee vielä ladata opetus data kansioineen Google Driveen. Voit joko tuoda kansiot itse tai käyttää tämän repon valmista mallipohjaa. Kansion nimi on 'Dataset' ja se sisältää kaikki tarvittavat tiedostot mallin opettamiseen.

</details>

## Tiedostojen lataaminen ja toiminnan varmistaminen

<details>

<summary>Ohjeet</summary>

1. Varmista että Google Drive kansion polku on vastaavanlainen.

```
Google Drive
└── oma-kansio
    └── Dataset
        └── oman-mallin-nimi
            ├── test
            ├── train
            ├── valid
            └── dataset.yaml
```

2. Voit jakaa kuvat ja annotaatiot siten, että 90-80% on train kansiossa ja 10-20% valid kansiossa

3. Varmista, että train kansio sisältää kuvat, niiden annotaatiot sekä classes.txt. Kuvien ja annotaatioiden nimien pitää olla samat. Katso myös, että dataset.yaml tiedosto on oikein ja oikeassa kansiossa.

```yaml
path: ../drive/MyDrive/oma-kansio/Dataset/oman-mallin-nimi
train: ../train/
val: ../valid/
test: ../test/

nc: 1 # varmista, että luokkien määrä on oikein
names: ["kissa"] # varmista, että luokkien nimet ovat oikein ja samassa järjestyksessä kuin classes.txt tiedostossa. Se löytyy train kansiosta.
```

4. Varmista että valid kansio sisältää kuvat ja niiden annotaatiot. Kuvien ja annotaatioiden nimien pitää olla samat.

5. Varmista että Colabin koodit ja polut ovat oikein.

```
!yolo task=detect mode=train model=yolov8n.pt data=../content/drive/MyDrive/oma-kansio/Dataset/oma-malli/dataset.yaml epochs=200 imgsz=640
```

</details>

## Tekoälymallin opettaminen(vihdoin)

<details>

<summary>Ohjeet</summary>

Nyt kun kaikki on valmista, voimme opettaa mallin. Tämä voi kestää jonkin aikaa, riippuen kuinka monta kuvaa on ja kuinka monta epochia käytetään.

Voit aloittaa mallin opettamise painamalla 'Suorituspalvelu' nappia ja sieltä 'Suorita kaikki'. tai voit myös painaa 'Ctrl + F9'.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/ceb9e08e-a3aa-42cf-ab66-e70d9a0c99c8)

Colabin konsolissa pitäisi näkyä jotain tällaista.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/91279152-1704-4141-aa39-6410102a7962)

Nyt vain odotetaan, että malli on opettanut tarpeeksi ja voimme testata sitä. Opettamasi mallin löydät täältä. train-kansion numero kasvaa joka kerta kun opetat mallia. Kansiossa oleva best.pt tiedosto on paras malli, jota voit käyttää.

Laita best.pt tiedosto samaan kansioon kuin [webcam_detect.py](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/blob/master/webcam_detect.py) tiedosto ja käynnistä se pythonilla. Se käyttää tietokoneeseen kytkettyä webkameraa ja tunnistaa kuvassa olevat objektit.

```
runs
└── detect
    └── train
        └── weights
            └── best.pt
```

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/709bb071-0a72-47d0-8a6d-0ec750c138e9)

</details>

## Tekoälymallin tilastojen tulkinta

<details>

<summary>Ohjeet</summary>

Tässä esimerkki mallissa opetin mallin tunnistamaan rajakytkimiä. Malli tunnistaa rajakytkimet hyvin, mutta se tunnistaa myös muita objekteja, joita ei pitäisi tunnistaa. Tämä johtuu siitä, että malli ei ole opettanut tarpeeksi. Tässä mallissa käytin vain 36 kuvaa, joten se ei ole tarpeeksi.

Huomasin myös että malli toimii paremmin jos pidän rajakytkintä kädessäni eli se "oppi tunnistamaan" jos kytkintä pidetään kädessä. Tämä johtuu siitä, että malli ei ole opettanut tarpeeksi ja tarpeeksi monipuolisilla kuvilla. Kaikki kuvat oli otettu rajakytkin kädessä.

train_batch.jpg tiedostot sisältää kuvan, jossa on mallin tunnistamat objektit ja niiden luokat.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/366f16a5-5d3f-48c6-bb47-7f8f89791c84)

val_batch.jpg tiedostot sisältää kuvan, jossa on mallin tunnistamat objektit ja niiden luokat.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/fe08b083-bd1a-4757-bd1b-424faddd84c2)

"Tärkeimpiä" tilastoja ovat:

-   mAP50 - Tämä kertoo miten hyvin malli tunnistaa kappaleeet yli 50% varmuudella.
-   mAP50-95 - Tämä kertoo miten hyvin malli tunnistaa kappaleeet 50-95% varmuudella.

Kuvaa katsottaessa voidaan huomata, että malli tunnistaa rajakytkimet yli 60% varmuudella jo 20 epochin jälkeen.

![kuva](https://github.com/JonneSaloranta/YOLO-tunnistus-ohjeet/assets/72470168/aa119816-6f41-467c-9411-629c39d6d798)

</details>
