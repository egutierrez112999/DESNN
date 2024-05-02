###Cryptanalysis of DES
Used a Neural Network to decrypt ciphertext using plaintext/ciphertext pairs. Used Python version of DES by https://github.com/twhiteman/pyDes to generate the ciphertext. The ciphertext was generated with the same key.
The Network takes as input the 64-bit ciphertext and uses the  64-bit plaintext to verify. The Network learns and the resulting model theoretically decyphers any ciphertext for the given key. This work is replicating a previous project as a form of verification. 

Notes: Clean up and Organize the Repo
