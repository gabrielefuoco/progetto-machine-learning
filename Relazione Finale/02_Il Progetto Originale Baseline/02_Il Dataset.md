# Il Dataset

Le caratteristiche tecniche del dataset utilizzato, inclusi la natura sintetica, lo sbilanciamento delle classi e la struttura delle feature eterogenee, sono determinanti per la strategia di rilevazione.

Il dataset impiegato è una derivazione del noto **PaySim** @lopez2016paysim, un simulatore di transazioni finanziarie mobili (Mobile Money) progettato per generare dati aggregati sintetici che rispecchiano fedelmente il comportamento di transazioni reali private, preservando la privacy degli utenti.

### Caratteristiche Quantitative
Il file `fraud_dataset_v2.csv` contiene un totale di **532.909 transazioni**. La distribuzione delle classi rivela uno sbilanciamento significativo, anche se mitigato rispetto ai dati grezzi originali per scopi sperimentali:

*   Transazioni Totali: ~532k
*   Frodi ("Malicious"): ~1% - 2% (variabile a seconda del subset di test)
*   Transazioni Lecite ("Regular"): >98%

### Struttura delle Feature
Le variabili includono attributi categorici ad alta cardinalità e valori numerici continui. Le feature principali (pre-elaborazione) sono:

*   **KTOSL** (Tipo di conto/transazione): Codice univoco che identifica la tipologia operativa.
*   **PRCTR** (Profit Center): Centro di profitto associato alla transazione (location geografica o unità business).
*   **BSCHL** (Posting Key): Chiave contabile che definisce la direzione del flusso (Debito/Credito).
*   **HKONT** (General Ledger Account): Il conto di contabilità generale impattato. Questa è la variabile categoriale più critica, con centinaia di valori unici possibili.
*   **WAERS** (Valuta): Codice valuta della transazione.
*   **BUKRS** (Company Code): Codice identificativo dell'entità legale.
*   **DMBTR** (Amount in Local Currency): Importo numerico in valuta locale.
*   **WRBTR** (Amount in Document Currency): Importo numerico nella valuta originale del documento.

La complessità principale risiede nella natura mista di questi dati. Mentre un modello umano può intuire che certi conti (`HKONT`) sono più rischiosi di altri, una rete neurale deve apprendere queste relazioni da zero. L'elevata cardinalità delle variabili categoriali impone sfide significative in fase di pre-elaborazione (discussa nel paragrafo successivo), poiché codificarle in modo ingenuo porterebbe a vettori di input sparsi di dimensioni ingestibili.
