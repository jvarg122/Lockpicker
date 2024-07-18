# Lockpicker
## Overview
Password cracking involves attempting to discover passwords for various accounts or systems, which can be both beneficial (e.g., recovering forgotten passwords) and malicious (e.g., unauthorized access). The objective of "Lockpicker" is to leverage GPUs instead of the CPU to accelerate password generation, thereby reducing the time required for brute-force.

$$\left( \sum_{i=1}^L C^{i} \right)$$  
The number of possible passwords where:  
`C` = number of legal characters  
`L` = maximum password length  

The testing model involves generating passwords of varying lengths (4 to 12 characters) using different character sets (digits, lowercase letters, Base32, Base64). These passwords are then subjected to cracking attempts using GPU parallelization until all possible combinations are exhausted and checked against the target password.

| Password  | ID | Index |
| ------------- | ------------- | ------------- |
| aaa | 0 | 000 |
| aab | 1 | 001 |
| bbb | 7 | 111 |

## Usage

**Installation**
1. Clone this repository:
```
git clone https://github.com/jvarg122/Lockpicker
```
2. Dependencies
- UC Riverside's EE server [Bender]() which has an NVIDIA 4 GeForce RTX 2070 GPU was used to write, compile, and run CUDA code.

To deploy and run the program, run ‘make’ and use the following command structure:

**Arguments**
1. Executable name
2. Password length
3. Preset or Custom Character Set (0-1)
- If preset is chosen, Choose from predefined sets of characters (0-3):
  - `0`: Digits (0-9)
  - `1`: Lowercase letters (a-z)
  - `2`: Base32
  - `3`: Base64
- If custom is chosen, input:
  - The number of custom characters
  - The characters as a string
4. Random or Custom Password (0-1)
    - If custom is chosen, enter the custom password to find
5. Print or do **NOT** print the generated passwords (0-1)
    - **Note:** If the number of generated passwords is in the thousands or more, printing can significantly slow down the program and make the output difficult to manage.

**Sample Input:**
```
./a.out 3 1 3 abc 1 ab 1
```
This example launches the program with a maximum password length of 3, valid characters {a, b, c}, and the target password set as "ab".

## Data

**Test Cases (Character Sets):**
| pswd length  | 0-9 | a-z | base32 | base 64 |
| ------------- | ------------- | ------------- | -------------| -------------|
| 4 | 1234 | asdf | A5DF | A5df |
| 6 | 123456 | qwerty | QW3RTY | Qw3R+y |
| 8 | 39707534 | computer | COMPUT3R | C0mPU+3r |
| 10 | 7684342856 | university | UNIV3R5ITY | uN1v3R51+y |
| 12 | 434699650223 | verylongpswd | V3RYLONGP5WD | V3rYL0ngP5wD |

**Password Cracking Time with GPU:**
| pswd length  | 0-9 | a-z | base32 | base 64 |
| ------------- | ------------- | ------------- | -------------| ------------- |
| 4 | 0.48 | 0.49 | 0.07 | 0.11 |
| 6 | 0.12 | 0.17 | 0.13 | 22.81 |
| 8 | 0.16 | 48.85 | 32.93 | ------------- |
| 10 | 4.05 | ------------- | ------------- | ------------- |
| 12 | 513.99 | ------------- | -------------| X* |

**Password Cracking Time with CPU:**
| pswd length  | 0-9 | a-z | base32 | base 64 |
| ------------- | ------------- | ------------- | -------------| ------------- |
| 4 | 0 | 0 | 0 | 0.03 |
| 6 | 0.01 | 3.66 | 82.03 | 1800.23 |
| 8 | 3.96 | 4539.77 | 15003.26 | ------------- |
| 10 | 1679.19 | ------------- | ------------- | ------------- |
| 12 | ------------- | ------------- | -------------| ------------- |

**Note:** dashed cells in the table indicate that tests took considerably longer than anticipated. It is reasonable to infer that these tests lasted over an hour.

## References
- Brian Gleeson, Proofpoint. ["5 Password Cracking Techniques Used in Cyber Attacks."](https://www.proofpoint.com/us/blog/information-protection/password-cracking-techniques-used-in-cyber-attacks)
