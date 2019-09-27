Steganografierea si extragerea mesajului steganografiat se realizeaza cu scripturile steganoEncode.py si respectiv steganoDecode.py.
___________________________________________________________
steganoEncode.py
Acest script e construit cu python3.
Accepta ca paramterii:
    -2 inputuri, precizate dupa numele scriptului
    -una din cele 3 optiuni (-t, -s si -c) pentru modul LSB, modul AM si modul derivat din LSB
    -un tag de output (-o) si numele fisierului de output
Pentru a rula corect avem sintaxa:
    python3 steganoEncode.py input1.extensie1 input2.extensie2 -x -o numeOutput.extensieOutput
    Unde x poate fi t sau s sau c, doar una dintre cele precizate;
De preferat: pentru a rula scriptul usor, fisierele cu care se lucreaza, (de preferabil), ar trebui sa fie in acelasi director

______________________________________________________
steganoDecode.py
Acest script e construit cu python3.
Accepta ca paramterii:
    -un input, precizat dupa numele scriptului
    -una din cele 3 optiuni (-t, -s si -c) pentru modul LSB, modul AM si modul derivat din LSB
    -un tag de output (-o) si numele fisierului de output
Pentru a rula corect avem sintaxa:
    python3 steganoDecode.py input.extensieInput -x -o numeOutput.extensieOutput
    Unde x poate fi t sau s sau c, doar una dintre cele precizate;
De preferat: pentru a rula scriptul usor, fisierele cu care se lucreaza, (de preferabil), ar trebui sa fie in acelasi director

Autor - Martinescu Sorin-Alexandru 332AA


