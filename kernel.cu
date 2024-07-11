#include <stdio.h>
#include <stdlib.h> 
#include <math.h>

#define BLOCK_SIZE 1024
#define GRID_SIZE 36

// Function to convert number to index array based on character set
__device__ void numToIndex(unsigned long long int num, int numChars, const int pswdLen, int* arr) { 
    int iterator = pswdLen - 1;

    while(iterator >= 0)
    {
        unsigned long long int val = num % numChars;
        arr[iterator] = val;

        num /= numChars;
        iterator--;
    }
}

__device__ bool checkPassword(char* c1, char* c2, const int pswdLen) 
{
    int c1Len = 0;
    int c2Len = 0;

    // Calculate the length of the first and second password
    for(int i = 0; c1[i] != '\0'; i++) { c1Len++; }
    for(int i = 0; c2[i] != '\0'; i++) { c2Len++; }

    if(c1Len != c2Len) return false;

    for(int i = 0; i < pswdLen; i++)  
    {
        if(c1[i] != c2[i]) // If lengths are not equal, passwords cannot be equal
        {
            return false;
        }
    }

    return true;
}

__device__ void incrementIndex(int* index, int pswdLen, int base) {
    for(int i = pswdLen - 1; i >= 0; i--)
    {
        if(index[i] == (base - 1))
        {
            index[i] = 0; 
        }

        else
        {
            index[i] += 1; return; 
        }
    }
}

// Create password from index array and character set
__device__ void createPasswordFromIndex(const int* index, const char* characterSet, int pswdLen, char* password) { 
    int i = 0;
    
    for (i = i; i < pswdLen; i++) {
        password[i] = characterSet[index[i]];
    }
    password[i] = '\0';
}


// Kernel function to perform password cracking         
__global__ void crack_kernel(unsigned long long int totalPswds, char* validChars, int numValidChars, const int pswdLen, char* password, bool* doneness, bool* printPasswords) {
    bool localPrintPasswords = *printPasswords; 
    int numThreads = BLOCK_SIZE * GRID_SIZE;
    unsigned long long int workPerThread = totalPswds / numThreads; 
    unsigned long long int overhead = totalPswds % numThreads; 

    // Calculate the starting and ending numbers for the range of passwords each thread will handle
    unsigned long long int startingNum, endingNum;
    startingNum = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * workPerThread;
    if(workPerThread == 0) { endingNum = 0; }
    else { endingNum = startingNum + workPerThread - 1; }

    int startingIndex[64];
    int endingIndex[64];

    char p[64];

    if(workPerThread > 0) // Process range of passwords for each thread
    {
        numToIndex(startingNum, numValidChars, pswdLen, startingIndex);
        numToIndex(endingNum, numValidChars, pswdLen, endingIndex);
        
        while(startingNum < endingNum && !(*doneness)) { // replace condition later
            // Generate password from index
            createPasswordFromIndex(startingIndex, validChars, pswdLen, p);
            if(localPrintPasswords) printf("%s", p);

            // Check if generated password matches target password
            if(checkPassword(p, password, pswdLen)) {
                *doneness = true; // Set flag indicating password is found
            }

            startingNum++;
            incrementIndex(startingIndex, pswdLen, numValidChars);
        }
    }

    // Handle remaining passwords if there's an overhead
    if(!(*doneness) && (blockIdx.x * BLOCK_SIZE + threadIdx.x < overhead))
    {
        int passwordNum = GRID_SIZE * BLOCK_SIZE * workPerThread + (blockIdx.x * BLOCK_SIZE + threadIdx.x);
        int index[64];

        numToIndex(passwordNum, numValidChars, pswdLen, index);
        createPasswordFromIndex(index, validChars, pswdLen, p);

        if(localPrintPasswords) printf("%s\n", p);

        if(checkPassword(p, password, pswdLen))
        {
            *doneness = true; // Set flag indicating password is found
        }
    }
}

void crack(char* validChars, int numValidChars, const int pswdLen, char* password, int printPasswords, bool* device_isDone, bool* device_printPasswords) {
    dim3 blockSize (1024, 1, 1); // CUDA block size (1024 threads per block)
    dim3 gridSize (36, 1, 1); // CUDA grid size (36 blocks)
    unsigned long long int totalPswds = pow(numValidChars, pswdLen);

    crack_kernel<<<gridSize, blockSize>>>(totalPswds, validChars, numValidChars, pswdLen, password, device_isDone, device_printPasswords);
}