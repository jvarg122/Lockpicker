#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <time.h> 
#include "kernel.cu"

// Generate a random password from a given character set
void generatePassword(char* password, char* characterList, int length) {
    int numChar = strlen(characterList);
    
    srand(time(0));

    int i;
    for (i = 0; i < length; ++i) {
        int randomIndex = rand() % numChar;
        password[i] = characterList[randomIndex];
    }
    password[i] = '\0'; // Null-terminate the password string
}

int main(int argc, char *argv[])
{
    char* validChars = (char*)malloc(256); 
    char* presetPassword = (char*)malloc(256); 
    int presetPasswordLength;

    int arg = 1;

    int pswdLen = atoi(argv[arg]); arg++; // Password length from command line argument
    int runType = atoi(argv[arg]); arg++; // Type of character set or custom input
    int numValidChars;

    // Determine character set based on runType
    '''
    The preset character sets are as follows:

    case 0: digits 0-9
    case 1: lowercase letters (a-z)
    case 2: base32
    case 3: base64
    '''
    switch(runType) 
    {
        case 0: 
        {
            int charSet = atoi(argv[arg]); arg++;

            switch(charSet)
            {
                case 0: 
                {
                    numValidChars = 10;
                    const char* str = "0123456789";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }

                case 1: 
                {
                    numValidChars = 26;
                    const char* str = "abcdefghijklmnopqrstuvwxyz";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }

                case 2: 
                {
                    numValidChars = 32;
                    const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }

                case 3: 
                {
                    numValidChars = 64;
                    const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }
                    break;
                }

                default:
                {
                    printf("unrecognized argument at %d: pick a character preset: 0 = decimal numbers, 1 = lowercase letters, 2 = base32, 3 = base64\n", arg);
                    exit(-1);
                    break;
                }
            }
            break;
        }
        case 1: // Custom character set provided
        {
            numValidChars = atoi(argv[arg]); arg++;
            const char* str = argv[arg]; arg++;
            for(int i = 0; i < numValidChars; i++) { validChars[i] = str[i]; }

            break;
        }

        default:
        {
            printf("unrecognized argument at %d: pick a preset (0) or enter your own (1)\n", arg);
            exit(-1);
            break;
        }
    }

    int hasGivenPassword = atoi(argv[arg]); arg++;
    
    switch(hasGivenPassword)
    {
        case 0: // Generate a random password
        {
            generatePassword(presetPassword, validChars, pswdLen);
            presetPasswordLength = pswdLen;

            break;
        }

        case 1: // Use provided preset password
        {
            const char* argPswd = argv[arg]; arg++;
            presetPasswordLength = strlen(argPswd);

            for(int i = 0; i < presetPasswordLength; i++) { presetPassword[i] = argPswd[i]; }            
            break;
        }

        default:
        {
            printf("unrecognized argument at %d: generate a random password (0) or enter your own (1)\n", arg);
            exit(-1);
            break;
        }
    }

    // Determine if passwords should be printed during cracking
    int printPswds = atoi(argv[arg]); arg++; 
    bool* d_printPswds; cudaMalloc(&d_printPswds, sizeof(bool));

    switch(printPswds)
    {
        case 0: { cudaMemset(d_printPswds, false, sizeof(bool)); break;}
        case 1: { cudaMemset(d_printPswds, true, sizeof(bool));  break;}

        default:
        {
            printf("unrecognized argument at %d: don't print passwords (0) or print passwords (1)\n", arg);
            exit(-1);
            break;
        }
    }

    char* d_validChars;
    char* d_presetPassword;

    cudaMalloc((void**)&d_validChars, numValidChars * sizeof(char));
    cudaMalloc((void**)&d_presetPassword, presetPasswordLength * sizeof(char));
    cudaMemcpy(d_validChars, validChars, numValidChars * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_presetPassword, presetPassword, presetPasswordLength * sizeof(char), cudaMemcpyHostToDevice);

    bool h_isDone = false;
    bool* d_isDone; cudaMalloc(&d_isDone, sizeof(bool)); cudaMemset(d_isDone, false, sizeof(bool));

    int iterator = 1;

    clock_t start, foundPswdTime;

    start = foundPswdTime = 0;

    start = clock();
    while(!h_isDone && iterator <= pswdLen)
    {
        printf("launching kernel w/ pswdLen of %d\n", iterator);

        crack(d_validChars, numValidChars, iterator, d_presetPassword, printPswds, d_isDone, d_printPswds);

        cudaError_t cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) printf("Unable to launch kernel\n");

        cuda_ret = cudaMemcpy(&h_isDone, d_isDone, sizeof(bool), cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) printf("failed to copy isDone from kernel\n");
        if(h_isDone)
        {
            printf("password was found.\n");
            foundPswdTime = clock();
            break;
        }

        iterator++;        
    }

    // Calculate elapsed time
    double timeElapsed = ((double)(foundPswdTime - start)) / CLOCKS_PER_SEC;
    printf("found pswd in %es\n", timeElapsed);
    printf("start: %f, foundPswdTime: %f\n", (double)start, (double)foundPswdTime);

    // Free allocated memory on GPU
    cudaFree(d_validChars);
    cudaFree(d_presetPassword);
    cudaFree(d_isDone);
    cudaFree(d_printPswds);

    return 0;
}