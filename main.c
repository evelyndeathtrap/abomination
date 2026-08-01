#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef void* (*GraphFunc)(void*, void*, int);

// Function prototypes for graph node operations
void* func_add(void* a_ptr, void* b_ptr, int size);
void* func_mul(void* a_ptr, void* b_ptr, int size);
void* func_trig(void* a_ptr, void* b_ptr, int size);
void* func_div(void* a_ptr, void* b_ptr, int size);
void* func_poly(void* a_ptr, void* b_ptr, int size);
void* func_tanh(void* a_ptr, void* b_ptr, int size);
void* func_log(void* a_ptr, void* b_ptr, int size);
void* func_deriv(void* a_ptr, void* b_ptr, int size);
void* func_max(void* a_ptr, void* b_ptr, int size);
void* func_sigmoid(void* a_ptr, void* b_ptr, int size);

typedef struct {
    int id;
    double bias;
    double gate;
    GraphFunc* functionLibrary;
    int funcCount;
    int funcCapacity;
    GraphFunc activeFunc;
    double** weights; // Dynamic 2D matrix: weights[totalNodes][vectorSize]
    double* output;   // Dynamic vector of size vectorSize
} GraphArithmeticNode;

typedef struct {
    int vectorSize;
    int totalNodes;
    int nodeCapacity;
    GraphArithmeticNode* nodes; // Dynamic array of nodes
    int* inputNodeIds;
    int* outputNodeIds;
} FullyInterconnectedGraphNetwork;

// Helper allocation functions for dynamically growing arrays
double* create_zero_vector(int size) {
    double* v = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) v[i] = 0.0;
    return v;
}

void* func_add(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = a[i] + b[i];
    return res;
}

void* func_mul(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = a[i] * (b[i] != 0 ? b[i] : 1.0) - b[i];
    return res;
}

void* func_trig(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = sin(a[i]) * cos(b[i]);
    return res;
}

void* func_div(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = a[i] / (fabs(b[i]) + 1e-6);
    return res;
}

void* func_poly(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = pow(fabs(a[i]), 2.0) + pow(b[i], 2.0);
    return res;
}

void* func_tanh(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = tanh(a[i]) * (b[i] != 0 ? b[i] : 1.0);
    return res;
}

void* func_log(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = log(fabs(a[i]) + 1e-5) * exp(-fabs(b[i]));
    return res;
}

void* func_deriv(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = (sin(a[i] + b[i]) - sin(a[i])) / 1e-3;
    return res;
}

void* func_max(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = (a[i] > b[i]) ? a[i] : b[i];
    return res;
}

void* func_sigmoid(void* a_ptr, void* b_ptr, int size) {
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) res[i] = 1.0 / (1.0 + exp(-a[i] * b[i]));
    return res;
}

void init_node(GraphArithmeticNode* node, int id, int totalNodes, int vectorSize) {
    node->id = id;
    node->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    node->gate = (double)rand() / RAND_MAX;
    
    // Dynamically expand function capacity (infinitely extensible)
    node->funcCapacity = 32;
    node->funcCount = 10;
    node->functionLibrary = (GraphFunc*)malloc(sizeof(GraphFunc) * node->funcCapacity);
    
    node->functionLibrary[0] = func_add;
    node->functionLibrary[1] = func_mul;
    node->functionLibrary[2] = func_trig;
    node->functionLibrary[3] = func_div;
    node->functionLibrary[4] = func_poly;
    node->functionLibrary[5] = func_tanh;
    node->functionLibrary[6] = func_log;
    node->functionLibrary[7] = func_deriv;
    node->functionLibrary[8] = func_max;
    node->functionLibrary[9] = func_sigmoid;
    
    node->activeFunc = node->functionLibrary[rand() % node->funcCount];
    
    // Allocate dynamic weights matrix [totalNodes x vectorSize]
    node->weights = (double**)malloc(sizeof(double*) * totalNodes);
    for (int i = 0; i < totalNodes; i++) {
        node->weights[i] = (double*)malloc(sizeof(double) * vectorSize);
        for (int j = 0; j < vectorSize; j++) {
            node->weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    
    node->output = create_zero_vector(vectorSize);
}

void add_function_to_node(GraphArithmeticNode* node, GraphFunc newFunc) {
    if (node->funcCount >= node->funcCapacity) {
        node->funcCapacity *= 2;
        node->functionLibrary = (GraphFunc*)realloc(node->functionLibrary, sizeof(GraphFunc) * node->funcCapacity);
    }
    node->functionLibrary[node->funcCount++] = newFunc;
}

void compute_node(GraphArithmeticNode* node, double** nodeOutputs, int totalNodes, int vectorSize) {
    double* acc = create_zero_vector(vectorSize);
    for (int i = 0; i < vectorSize; i++) {
        acc[i] = node->bias;
    }
    
    for (int srcId = 0; srcId < totalNodes; srcId++) {
        if (srcId == node->id) continue;
        for (int i = 0; i < vectorSize; i++) {
            acc[i] += nodeOutputs[srcId][i] * node->weights[srcId][i];
        }
    }
    
    double* biasArr = create_zero_vector(vectorSize);
    for (int i = 0; i < vectorSize; i++) biasArr[i] = node->bias;
    
    double* funcRes = (double*)node->activeFunc(acc, biasArr, vectorSize);
    for (int i = 0; i < vectorSize; i++) {
        node->output[i] = funcRes[i] * node->gate;
    }
    
    free(acc);
    free(biasArr);
    free(funcRes);
}

void mutate_node(GraphArithmeticNode* node, int totalNodes, int vectorSize) {
    double randVal = (double)rand() / RAND_MAX;
    if (randVal < 0.35) {
        node->activeFunc = node->functionLibrary[rand() % node->funcCount];
    } else if (randVal < 0.60) {
        node->bias += (((double)rand() / RAND_MAX) * 0.5 - 0.25);
    } else if (randVal < 0.75) {
        node->gate += (((double)rand() / RAND_MAX) * 0.2 - 0.1);
    } else {
        for (int i = 0; i < totalNodes; i++) {
            for (int j = 0; j < vectorSize; j++) {
                node->weights[i][j] += (((double)rand() / RAND_MAX) * 0.5 - 0.25);
            }
        }
    }
}

void init_network(FullyInterconnectedGraphNetwork* net, int initialVectorSize, int initialTotalNodes) {
    net->vectorSize = initialVectorSize;
    net->totalNodes = initialTotalNodes;
    net->nodeCapacity = initialTotalNodes * 2;
    
    net->nodes = (GraphArithmeticNode*)malloc(sizeof(GraphArithmeticNode) * net->nodeCapacity);
    for (int i = 0; i < net->totalNodes; i++) {
        init_node(&net->nodes[i], i, net->totalNodes, net->vectorSize);
    }
    
    net->inputNodeIds = (int*)malloc(sizeof(int) * 2);
    net->inputNodeIds[0] = 0;
    net->inputNodeIds[1] = 1;
    
    net->outputNodeIds = (int*)malloc(sizeof(int) * 1);
    net->outputNodeIds[0] = net->totalNodes - 1;
}

void add_node_to_network(FullyInterconnectedGraphNetwork* net) {
    if (net->totalNodes >= net->nodeCapacity) {
        net->nodeCapacity *= 2;
        net->nodes = (GraphArithmeticNode*)realloc(net->nodes, sizeof(GraphArithmeticNode) * net->nodeCapacity);
        
        // Reallocate weight matrices for all existing nodes to accommodate the new totalNodes size
        for (int i = 0; i < net->totalNodes; i++) {
            net->nodes[i].weights = (double**)realloc(net->nodes[i].weights, sizeof(double*) * net->nodeCapacity);
            for (int j = net->totalNodes; j < net->nodeCapacity; j++) {
                net->nodes[i].weights[j] = (double*)malloc(sizeof(double) * net->vectorSize);
                for (int v = 0; v < net->vectorSize; v++) {
                    net->nodes[i].weights[j][v] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                }
            }
        }
    }
    
    int newId = net->totalNodes;
    init_node(&net->nodes[newId], newId, net->nodeCapacity, net->vectorSize);
    net->totalNodes++;
}

void forward_network(FullyInterconnectedGraphNetwork* net, double* inputVector1, double* inputVector2, double* outputResult) {
    // Dynamically allocate nodeOutputs matrix [totalNodes x vectorSize]
    double** nodeOutputs = (double**)malloc(sizeof(double*) * net->totalNodes);
    for (int i = 0; i < net->totalNodes; i++) {
        nodeOutputs[i] = create_zero_vector(net->vectorSize);
    }
    
    for (int i = 0; i < net->vectorSize; i++) {
        nodeOutputs[net->inputNodeIds[0]][i] = inputVector1[i];
        nodeOutputs[net->inputNodeIds[1]][i] = inputVector2[i];
    }
    
    for (int step = 0; step < 3; step++) {
        for (int i = 0; i < net->totalNodes; i++) {
            if (i == net->inputNodeIds[0] || i == net->inputNodeIds[1]) continue;
            compute_node(&net->nodes[i], nodeOutputs, net->totalNodes, net->vectorSize);
            for (int j = 0; j < net->vectorSize; j++) {
                nodeOutputs[i][j] = net->nodes[i].output[j];
            }
        }
    }
    
    int outId = net->outputNodeIds[0];
    compute_node(&net->nodes[outId], nodeOutputs, net->totalNodes, net->vectorSize);
    for (int j = 0; j < net->vectorSize; j++) {
        outputResult[j] = net->nodes[outId].output[j];
    }
    
    for (int i = 0; i < net->totalNodes; i++) {
        free(nodeOutputs[i]);
    }
    free(nodeOutputs);
}

void train_from_file(FullyInterconnectedGraphNetwork* net, const char* filePath, int maxEpochs) {
    FILE* f = fopen(filePath, "r");
    if (!f) {
        printf("Error: Training file \"%s\" not found.\n", filePath);
        exit(1);
    }
    
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* rawText = (char*)malloc(length + 1);
    fread(rawText, 1, length, f);
    fclose(f);
    rawText[length] = '\0';
    
    printf("Loaded array text file. Length: %ld chars. Infinite scale graph training initializing...\n", length);
    
    int datasetSize = length - net->vectorSize;
    if (datasetSize <= 0) {
        printf("Error: File content too short for vector size.\n");
        free(rawText);
        exit(1);
    }
    
    for (int epoch = 1; epoch <= maxEpochs; epoch++) {
        double totalError = 0.0;
        
        for (int i = 0; i < datasetSize; i++) {
            double* vectorIn1 = create_zero_vector(net->vectorSize);
            double* vectorIn2 = create_zero_vector(net->vectorSize);
            double* vectorTarget = create_zero_vector(net->vectorSize);
            
            for (int j = 0; j < net->vectorSize; j++) {
                vectorIn1[j] = (double)(unsigned char)rawText[i + j] / 255.0;
                vectorIn2[j] = (double)(unsigned char)rawText[i + j + 1] / 255.0;
                vectorTarget[j] = (double)(unsigned char)rawText[i + j + net->vectorSize] / 255.0;
            }
            
            double* preds = create_zero_vector(net->vectorSize);
            forward_network(net, vectorIn1, vectorIn2, preds);
            
            double sampleError = 0.0;
            for (int k = 0; k < net->vectorSize; k++) {
                double diff = vectorTarget[k] - preds[k];
                sampleError += diff * diff;
            }
            totalError += sampleError;
            
            for (int n = 0; n < net->totalNodes; n++) {
                net->nodes[n].gate += 0.005 * 0.01;
                for (int src = 0; src < net->totalNodes; src++) {
                    for (int v = 0; v < net->vectorSize; v++) {
                        net->nodes[n].weights[src][v] += 0.005 * 0.01;
                    }
                }
            }
            
            free(vectorIn1);
            free(vectorIn2);
            free(vectorTarget);
            free(preds);
        }
        
        double currentLoss = totalError / datasetSize;
        if (epoch % (maxEpochs / 5 == 0 ? 1 : maxEpochs / 5) == 0 || epoch == 1) {
            printf("Epoch %d/%d | Infinite Graph Loss: %f | Nodes: %d\n", epoch, maxEpochs, currentLoss, net->totalNodes);
        }
        
        if (currentLoss < 0.0001) {
            printf("\n[Early Stopping]: Reached target threshold at epoch %d.\n", epoch);
            break;
        }
        
        for (int n = 0; n < net->totalNodes; n++) {
            mutate_node(&net->nodes[n], net->totalNodes, net->vectorSize);
        }
        
        // Infinitely scale graph structure by expanding nodes periodically
        if (((double)rand() / RAND_MAX) < 0.20) {
            add_node_to_network(net);
        }
    }
    
    free(rawText);
    printf("Training successfully completed.\n");
}

void generate_text(FullyInterconnectedGraphNetwork* net, const char* seedText, int length) {
    double* v1 = create_zero_vector(net->vectorSize);
    double* v2 = create_zero_vector(net->vectorSize);
    
    for (int i = 0; i < net->vectorSize; i++) {
        v1[i] = 32.0 / 255.0;
        v2[i] = 32.0 / 255.0;
    }
    
    int seedLen = strlen(seedText);
    for (int i = 0; i < seedLen && i < net->vectorSize; i++) {
        v1[i] = (double)(unsigned char)seedText[i] / 255.0;
        v2[i] = (i + 1 < seedLen) ? (double)(unsigned char)seedText[i + 1] / 255.0 : 32.0 / 255.0;
    }
    
    printf("\n--- Infinite Scale Graph Text Generation Output ---\n");
    printf("%s", seedText);
    
    for (int step = 0; step < length; step++) {
        double* targetVec = create_zero_vector(net->vectorSize);
        forward_network(net, v1, v2, targetVec);
        
        int charCode = (int)round(targetVec[0] * 255.0);
        if (charCode < 32) charCode = 32;
        if (charCode > 126) charCode = 126;
        
        putchar(charCode);
        
        for (int i = 0; i < net->vectorSize; i++) {
            v1[i] = v2[i];
            v2[i] = targetVec[i];
        }
        free(targetVec);
    }
    printf("\n");
    
    free(v1);
    free(v2);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));
    FullyInterconnectedGraphNetwork net;
    
    // Initialize with extensible dynamic scale dimensions
    init_network(&net, 8, 12);
    
    if (argc >= 3 && strcmp(argv[1], "--train") == 0) {
        train_from_file(&net, argv[2], 200);
    } else {
        const char* seed = "Infinite";
        generate_text(&net, seed, 60);
    }
    
    return 0;
}
