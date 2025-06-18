# Implementarea si optimizarea unui brat robotic controlat prin inteligenta artificiala pentru manipularea pieselor de sah
## Introducere
Proiectul foloseste C++ si Python pentru a crea un ecosistem de joc pentru un brat fizic ce foloseste inteligenta artificiala pentru a-si decide, detecta si muta piesele.

## Pasi pentru rularea codului
1. Placa arduino trebuie conectata la masina ce ruleaza codul python. (Sursa de alimentare a proiectului NU trebuie conectata inca la alimentare)
2. (OPTIONAL) Se poate rula codul Arduino arduino/test_sensors/test_sensors/test_sensors.ino pentru a verifica daca sezorii reed switch merg corespunzator din serial monitor.
3. Se incarca pe placa arduino codul din arduino/data_transfer/data_transfer.ino si se asigura faptul ca nu este deschis niciun serial monitor (codul Python va esua in acest caz).
4. Se ruleaza codul Python data_transfer.py si se urmareste in terminal rularea fara probleme a acestuia.
5. Se conecteaza sursa de alimentare a proiectului la curent.
6. (OPTIONAL) Se pot modifica date din rubrica "SETARI AJUSTABILE" in settings.py pentru ajustarea dificultatii, ai-ului folosit sau schimbarea in modul de testare a codului.
7. Jocul incepe.
