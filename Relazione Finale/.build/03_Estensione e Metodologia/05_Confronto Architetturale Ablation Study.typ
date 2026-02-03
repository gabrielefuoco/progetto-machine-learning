= Confronto Architetturale (Ablation Study)
<confronto-architetturale-ablation-study>
In questo paragrafo conclusivo del capitolo metodologico, si introduce
l'esperimento comparativo volto a isolare e quantificare il contributo
specifico della componente avversaria dell'AAE.

=== La Domanda di Ricerca
<la-domanda-di-ricerca>
L'Adversarial Autoencoder è un'architettura indubbiamente elegante, ma
introduce una notevole complessità computazionale e strutturale (tre
reti neurali, training in due fasi, instabilità minimax). La domanda
fondamentale è: #strong[Tutta questa complessità è giustificata dai
risultati?] Il paper originale confronta l'AAE con tecniche classiche
(PCA) o dati grezzi, ma non fornisce un confronto diretto con un
Autoencoder Standard (puro Deep Learning non regolato) sulla stessa
architettura.

=== Design dell'Ablation Study
<design-dellablation-study>
Per colmare questa lacuna, abbiamo implementato una versione "ablata"
del modello (`train_standard_ae.py`): \* #strong[Architettura Identica];:
Lo Standard AE mantiene esattamente la stessa struttura dell'Encoder e
del Decoder dell'AAE (Stessi layer, neuroni, funzioni di attivazione
LeakyReLU). \* #strong[Rimozione del Discriminatore];: Viene eliminata
totalmente la rete Discriminatrice e la fase di "Regularization". \*
#strong[Loss Function Semplificata];: L'addestramento è guidato
esclusivamente dalla minimizzazione dell'errore di ricostruzione
($L_(r e c o n)^2$).

Questo setup garantisce che qualsiasi differenza di performance (MAUC)
misurata successivamente sia attribuibile #emph[unicamente] all'effetto
della regolarizzazione avversaria sullo spazio latente, e non a
differenze nella capacità della rete neurale. Se l'AAE è superiore, deve
esserlo perché la forma della distribuzione latente (Gaussiana) facilita
la classificazione. Se invece lo Standard AE (che lascia i dati liberi
di organizzarsi nello spazio latente) performa meglio, significa che la
regolarizzazione imposta dall'AAE potrebbe essere controproducente per
questo specifico dataset.
