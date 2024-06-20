# Raccomandazione agricola intelligente(ICON 23/24)
Repository caso di studio per ilcorso di Ingegneria della conoscenza, A.A 2023-2024, studenti

- Campanozzi Antonio (mat **763605**)
- Dimauro Vito Mauro (mat **738719**)

## Installazioni:

- installa Protegé:


    https://protege.stanford.edu/
- installa SWI Prolog (**il progetto usa la versione 9.0.4**):

    https://www.swi-prolog.org/download/stable?show=all

- Il codice fa uso di alcuni elementi aggiunti nella versione **3.10** di python, per via di alcuni piccoli dettagli (*f-strings*) abbiamo constatato che il programma non da problemi con un interprete versine **3.12**:

    https://www.python.org/downloads/release/python-3120/

## Avvio del sistema
- clona la repository:

    ```
    git clone https://github.com/AntonioCampanozzi/ICON_23-24.git
    ```
- posizionati nella directory:
    
    ```
    cd ICON_23-24
    ```
- crea l'ambiente virtuale(**facoltativo**):

    ```
    python -m venv ICON_23-24
    ```
- se creato l'ambiente, attivalo (**esempio su Windows**):
    ```
    ICON_23-24/Scripts/activate
    ```
    **N.B: best practice è disattivare l'ambiente virtuale, su Windows è possibile farlo scrivendo *deactivate* sul terminale**

- Installare le dipendenze:

    ```
    pip install -r requirements.txt
    ```

## Utilizzo

dopo aver avviato il main:

```
python main.py
```

verranno mostrate tre opzioni:

- consultare l'ontologia
- predire una coltura in base alle caratteristiche inserite
- interrogare la knowledge base prolog

L'ontologia consultata è **agricoltura.owl** presente nella directory *ontology*, per le predizioni viene utilizzata la random forest presente in *trained models*, la knowledge base è **crop.py** presente in *kb*.




   
