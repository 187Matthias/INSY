# Neuronales Netz from scratch (Ziel: XOR lernen)

## Projektziel
Ich baue ein eigenes, voll funktionsfähiges neuronales Netz in Python ohne ML-Frameworks (nur `numpy` und die Standardbibliothek).
Das Netz soll am Ende XOR lernen, speicherbar/ladbar sein und ein echtes, nachvollziehbares Netz sein.

## Aufgabenstellung / Unterlage
- Die Aufgabe steht in `Neural Network/4AHITS_NN_from_scratch.pdf`. Sie ist die Referenz dafür, was gemacht werden soll.
- Bei Unklarheiten zuerst im PDF nachlesen, dann erst nachfragen.
- Schrittweise vorgehen, in der Reihenfolge des PDFs (Part 1 → Part 6). Aktuell relevant: **Part 1 und Part 2**.

## Netz-Spezifikation (aus dem PDF)
- Klassen: `Neuron`, `Layer`, `NeuralNetwork`
- `Neuron`: Gewichte, Bias, Aktivierung (Standard: Sigmoid), speichert letzte Inputs und Output, `update()` für Gewichte/Bias
- `Layer`: Liste von Neuronen, `forward(inputs)`, `backward(errors, learning_rate)`, gibt Fehler an vorherige Schicht zurück
- `NeuralNetwork`: Liste von Layern, `train(inputs, targets)`, `predict(inputs)`, `dump_json()`, `load()`, `dump_txt()`, `dump_initfile()` (.h)
- Kostenfunktion: MSE, J = ½ (ŷ − y)²
- Training: Stochastic Gradient Descent (ein Sample pro Iteration)
- Sigmoid-Ableitung: `a * (1 - a)`
- Gewichts-Update: `W_new = W_old - alpha * dJ/dW`

## XOR-Ziel
- Architektur: 2 Input, 2 Hidden, 1 Output (Sigmoid)
- Epochen: 20000, Lernrate: 0.1
- Daten: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
- Danach: als `.json` exportieren, wieder laden, gleiche Vorhersagen prüfen

## Projektstruktur
Der Hauptordner `INSY` ist das Git-Repository (GitHub `187Matthias/INSY`). Jedes Projekt hat einen eigenen Ordner mit einer `DOKUMENTATION.qmd`; die Startseite zeigt automatisch eine Karte pro Projekt.
```
CLAUDE.md                          Regeln für alle Projekte
index.qmd                          Startseite (Projektkarten)
_quarto.yml                        Einstellungen der Website
.github/workflows/pages.yml        veröffentlicht die Website auf GitHub Pages
.venv/                             gemeinsame Python-Umgebung
Neural Network/                    Projekt "Neuronales Netz from scratch"
  DOKUMENTATION.qmd                Doku dieses Projekts (Projektseite)
  4AHITS_NN_from_scratch.pdf       Aufgabenstellung (nicht ändern)
  nn/                              Netz-Code (Neuron, Layer, NeuralNetwork)
  train_xor.py, train_7seg.py      trainieren + exportieren
  test_*.py                        Laden + Testen
  models/                          exportierte .json / .txt / .h Dateien
  web/                             NN-Werkbank (Website zum Ausprobieren)
```

## Dokumentation / Fortschritt
- Die Dokumentation des Projekts steht in `Neural Network/DOKUMENTATION.qmd` (Quarto, wird mit Git verwaltet und über GitHub Pages veröffentlicht) und zeigt, wie weit das Projekt aktuell ist (was fertig ist, was noch offen ist).
- Nach jedem abgeschlossenen Schritt `Neural Network/DOKUMENTATION.qmd` aktualisieren (Änderungen, Testergebnisse, offene Punkte) und im Hauptordner mit `.venv\Scripts\quarto.exe render` prüfen. Zum Einstieg in eine neue Sitzung zuerst dort nachlesen.
- Neues Projekt: neuer Ordner mit eigener `DOKUMENTATION.qmd` (mit `title`, `subtitle`, `description` im Kopf). Die Startseite muss dafür nicht geändert werden; neuer Eintrag in der Navigation (`_quarto.yml` und die Leiste in `Neural Network/web/index.html`).
- Design: Alle Seiten nutzen `assets/site.css` (Farben, Schriften, Navigation, wie in der NN-Werkbank). Design nur dort ändern, nicht pro Seite.

- Token-Verbrauch: In `Neural Network/DOKUMENTATION.qmd` (Abschnitt "Token-Verbrauch pro Prompt") steht, wie viele Tokens pro Prompt verbraucht wurden. Am Ende **jedes** Prompts eine neue Zeile eintragen (Prompt kurz beschreiben, ungefähre Tokens, Summe anpassen) und im Text kurz erwähnen. Die Werte sind Näherungen aus dem "tokens left"-Zähler (Differenz Anfang/Ende des Prompts).

## SICHERHEITSREGELN (wichtig!)
1. **Nichts löschen ohne Rückfrage.** Keine Datei, keinen Ordner und keinen größeren Codeabschnitt löschen oder überschreiben, ohne mich vorher zu fragen und ein klares Ja abzuwarten. Das gilt auch für `rm`, `del`, `git clean`, `git reset --hard` usw.
2. **Nur innerhalb dieses Projektordners arbeiten.** Kein Lesen, Schreiben, Anlegen oder Ausführen außerhalb dieses Ordners (keine `../`-Pfade, kein Home-Verzeichnis, keine Systemdateien).
3. Das PDF ist nur Lesematerial und wird nicht verändert.
4. Keine Pakete global installieren. Falls etwas nötig ist, zuerst fragen (am besten eine virtuelle Umgebung `.venv` im Projektordner).
5. Keine Befehle, die Netzwerkzugriffe oder Systemeinstellungen ändern, ohne Rückfrage.

## Arbeitsweise
- Vor größeren Änderungen kurz den Plan nennen und auf mein OK warten.
- Kleine Schritte: erst `forward`, dann `backward`, dann `train`, dann Export/Import.
- Code einfach und gut kommentiert halten (ich muss ihn verstehen und erklären können). Kommentare auf Deutsch.
- Erklären, *warum* etwas so gemacht wird (z. B. Kettenregel bei Backpropagation).
- Nach Änderungen das Training/den Test ausführen und das Ergebnis zeigen (Ausgabe der 4 XOR-Vorhersagen).
- Keine unnötigen Abhängigkeiten, keine ML-Frameworks (TensorFlow erst in Part 3, und nur zum Vergleich).

## Befehle
- Training: `python "Neural Network/train_xor.py"` (die Skripte wechseln selbst in ihren Ordner, funktioniert auch von dort mit `python train_xor.py`)
- Test: `python "Neural Network/test_xor.py"`
- Website rendern: `.venv\Scripts\quarto.exe render`