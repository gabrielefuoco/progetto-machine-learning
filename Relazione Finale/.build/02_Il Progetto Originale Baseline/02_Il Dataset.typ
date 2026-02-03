= Il Dataset
<il-dataset>
Questo paragrafo descrive le caratteristiche tecniche del dataset
utilizzato, evidenziandone la natura sintetica, lo sbilanciamento delle
classi e la struttura delle feature eterogenee.

Il dataset impiegato è una derivazione del noto #strong[PaySim]
#cite(<lopez2016paysim>, form: "prose");, un simulatore di transazioni
finanziarie mobili (Mobile Money) progettato per generare dati aggregati
sintetici che rispecchiano fedelmente il comportamento di transazioni
reali private, preservando la privacy degli utenti.

=== Caratteristiche Quantitative
<caratteristiche-quantitative>
Il file `fraud_dataset_v2.csv` contiene un totale di #strong[532.909
transazioni];. La distribuzione delle classi rivela uno sbilanciamento
significativo, anche se mitigato rispetto ai dati grezzi originali per
scopi sperimentali: \* Transazioni Totali: \~532k \* Frodi
("Malicious"): \~1% - 2% (variabile a seconda del subset di test) \*
Transazioni Lecite ("Regular"): \>98%

=== Struttura delle Feature
<struttura-delle-feature>
Le variabili includono attributi categorici ad alta cardinalità e valori
numerici continui. Le feature principali (pre-elaborazione) sono: \*
#strong[KTOSL] (Tipo di conto/transazione): Codice univoco che
identifica la tipologia operativa. \* #strong[PRCTR] (Profit Center):
Centro di profitto associato alla transazione (location geografica o
unità business). \* #strong[BSCHL] (Posting Key): Chiave contabile che
definisce la direzione del flusso (Debito/Credito). \* #strong[HKONT]
(General Ledger Account): Il conto di contabilità generale impattato.
Questa è la variabile categoriale più critica, con centinaia di valori
unici possibili. \* #strong[WAERS] (Valuta): Codice valuta della
transazione. \* #strong[BUKRS] (Company Code): Codice identificativo
dell'entità legale. \* #strong[DMBTR] (Amount in Local Currency):
Importo numerico in valuta locale. \* #strong[WRBTR] (Amount in Document
Currency): Importo numerico nella valuta originale del documento.

La complessità principale risiede nella natura mista di questi dati.
Mentre un modello umano può intuire che certi conti (`HKONT`) sono più
rischiosi di altri, una rete neurale deve apprendere queste relazioni da
zero. L'elevata cardinalità delle variabili categoriali impone sfide
significative in fase di pre-elaborazione (discussa nel paragrafo
successivo), poiché codificarle in modo ingenuo porterebbe a vettori di
input sparsi di dimensioni ingestibili.
