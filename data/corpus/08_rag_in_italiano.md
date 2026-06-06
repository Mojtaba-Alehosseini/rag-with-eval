# Recupero e Generazione Aumentata (RAG)

## Che cos'è il RAG
Il Retrieval-Augmented Generation (RAG) è una tecnica che àncora la risposta di un
modello linguistico ai documenti recuperati al momento della domanda. Il sistema prima
recupera i passaggi rilevanti da un corpus e poi genera la risposta basandosi su di essi,
invece di affidarsi soltanto alla conoscenza appresa durante l'addestramento.

## Perché usare il RAG
Il RAG riduce le allucinazioni perché la risposta è fondata su testo recuperato e può
citare le fonti, rendendo l'output verificabile. Inoltre permette di aggiornare la
conoscenza aggiornando il corpus, senza dover riaddestrare il modello.

## Le citazioni
Un sistema RAG affidabile allega una citazione a ogni affermazione, indicando il file di
origine e la pagina o la sezione. La regola è semplice: nessuna affermazione senza una
citazione. Quando il contesto recuperato non contiene la risposta, il sistema deve
dichiararlo esplicitamente.

## La valutazione
La fedeltà, in inglese faithfulness, misura se la risposta è supportata dal contesto
recuperato. La precisione del contesto, in inglese context precision, misura la qualità
del recupero. Il reranking con un cross-encoder migliora soprattutto la precisione del
contesto, perché porta i passaggi più rilevanti in cima alla lista.
