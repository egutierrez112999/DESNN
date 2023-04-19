import pyDes
from random import randint

def byte_to_bin(bytestuff):
    t_hex = bytestuff.hex()
    t_bin = bin(int(t_hex, 16)).zfill(8)
    return t_bin[2:]


def bin_to_list_of_ints(text):
    s = str(text)
    s_list = [i for i in s]
    s_list = [int(i) for i in s_list]
    while len(s_list) < 64:
        s_list.insert(0,0)
    return s_list


def create_text_pairs(des_object):
    r_num = randint(1000000000000000000, 9999999999999999999)
    r_byte = r_num.to_bytes(8, byteorder='big')
    ciphertext = des_object.encrypt(r_byte)
    ciphertext_list = bin_to_list_of_ints(byte_to_bin(ciphertext))
    decrypted_text = des_object.decrypt(ciphertext)
    decrypted_text_list = bin_to_list_of_ints(byte_to_bin(decrypted_text))
    plaintext_list = bin_to_list_of_ints(byte_to_bin(r_byte))
    return (ciphertext_list, plaintext_list)

def createpow2pairs(n, des_object):
    for i in range(2**n):
        print(create_text_pairs(des_object))

def check_correctness(des_object):
    pass

def main():
    des = pyDes.des('russross')
    createpow2pairs(13,des)

main()




        
#def main():
#    des = pyDes.des('russross')
#    plaintext = b'renquinn'
#    ciphertext = des.encrypt(plaintext)
#    print(des.getKey())
#    print(f"Encrypted: {ciphertext}")
#    print(f"Binary Repr: {bin_to_list_of_ints(byte_to_bin(ciphertext))}")
#    decrypted_text = des.decrypt(ciphertext)
#    print(f"Decrypted: {decrypted_text}")
#    print(f"Binary Repr: {bin_to_list_of_ints(byte_to_bin(decrypted_text))}")
