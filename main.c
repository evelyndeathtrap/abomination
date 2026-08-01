#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef void* (*GraphFunc)(void*, void*, int);

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
    double** weights; 
    double* output;   
} GraphArithmeticNode;

typedef struct {
    int vectorSize;
    int totalNodes;
    int nodeCapacity;
    GraphArithmeticNode* nodes; 
    int* inputNodeIds;
    int* outputNodeIds;
} FullyInterconnectedGraphNetwork;

double* create_zero_vector(int size) {
    if (size <= 0) return NULL;
    double* v = (double*)malloc(sizeof(double) * size);
    if (!v) {
        fprintf(stderr, "Memory allocation failed for vector of size %d\n", size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) v[i] = 0.0;
    return v;
}

void* func_add(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = a[i] + b[i];
    return res;
}

void* func_mul(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = a[i] * (b[i] != 0 ? b[i] : 1.0) - b[i];
    return res;
}

void* func_trig(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = sin(a[i]) * cos(b[i]);
    return res;
}

void* func_div(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = a[i] / (fabs(b[i]) + 1e-6);
    return res;
}

void* func_poly(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = pow(fabs(a[i]), 2.0) + pow(b[i], 2.0);
    return res;
}

void* func_tanh(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = tanh(a[i]) * (b[i] != 0 ? b[i] : 1.0);
    return res;
}

void* func_log(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = log(fabs(a[i]) + 1e-5) * exp(-fabs(b[i]));
    return res;
}

void* func_deriv(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = (sin(a[i] + b[i]) - sin(a[i])) / 1e-3;
    return res;
}

void* func_max(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = (a[i] > b[i]) ? a[i] : b[i];
    return res;
}

void* func_sigmoid(void* a_ptr, void* b_ptr, int size) {
    if (!a_ptr || !b_ptr || size <= 0) return NULL;
    double* a = (double*)a_ptr;
    double* b = (double*)b_ptr;
    double* res = create_zero_vector(size);
    for (int i = 0; i < size; i++) res[i] = 1.0 / (1.0 + exp(-a[i] * b[i]));
    return res;
}

void init_node(GraphArithmeticNode* node, int id, int totalNodes, int vectorSize) {
    node->id = id;
    node->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    node->gate = (double)rand() / RAND_MAX;
    
    node->funcCapacity = 32;
    node->funcCount = 10;
    node->functionLibrary = (GraphFunc*)malloc(sizeof(GraphFunc) * node->funcCapacity);
    if (!node->functionLibrary) {
        perror("Failed to allocate function library");
        exit(EXIT_FAILURE);
    }
    
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
    
    node->weights = (double**)malloc(sizeof(double*) * totalNodes);
    if (!node->weights) {
        perror("Failed to allocate weights matrix rows");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < totalNodes; i++) {
        node->weights[i] = create_zero_vector(vectorSize);
        for (int j = 0; j < vectorSize; j++) {
            node->weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    
    node->output = create_zero_vector(vectorSize);
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
    if (funcRes) {
        for (int i = 0; i < vectorSize; i++) {
            node->output[i] = funcRes[i] * node->gate;
        }
        free(funcRes);
    }
    
    free(acc);
    free(biasArr);
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
    if (!net->nodes) {
        perror("Failed to allocate network nodes");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < net->totalNodes; i++) {
        init_node(&net->nodes[i], i, net->nodeCapacity, net->vectorSize);
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
        GraphArithmeticNode* tempNodes = (GraphArithmeticNode*)realloc(net->nodes, sizeof(GraphArithmeticNode) * net->nodeCapacity);
        if (!tempNodes) {
            perror("Failed to reallocate network nodes");
            exit(EXIT_FAILURE);
        }
        net->nodes = tempNodes;
        
        for (int i = 0; i < net->totalNodes; i++) {
            double** tempWeights = (double**)realloc(net->nodes[i].weights, sizeof(double*) * net->nodeCapacity);
            if (!tempWeights) {
                perror("Failed to reallocate node weights");
                exit(EXIT_FAILURE);
            }
            net->nodes[i].weights = tempWeights;
            for (int j = net->totalNodes; j < net->nodeCapacity; j++) {
                net->nodes[i].weights[j] = create_zero_vector(net->vectorSize);
                for (int v = 0; v < net->vectorSize; v++) {
                    net->nodes[i].weights[j][v] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                }
            }
        }
    }
    
    int newId = net->totalNodes;
    init_node(&net->nodes[newId], newId, net->nodeCapacity, net->vectorSize);
    net->totalNodes++;
    net->outputNodeIds[0] = net->totalNodes - 1;
}

void forward_network(FullyInterconnectedGraphNetwork* net, double* inputVector1, double* inputVector2, double* outputResult) {
    double** nodeOutputs = (double**)malloc(sizeof(double*) * net->totalNodes);
    if (!nodeOutputs) {
        perror("Failed to allocate nodeOutputs");
        exit(EXIT_FAILURE);
    }
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
    if (!rawText) {
        perror("Failed to allocate rawText");
        fclose(f);
        exit(EXIT_FAILURE);
    }
    fread(rawText, 1, length, f);
    fclose(f);
    rawText[length] = '\0';
    
    printf("Loaded array text file. Length: %ld chars. Bounded memory graph training initializing...\n", length);
    
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
            printf("Epoch %d/%d | Graph Loss: %f | Nodes: %d\n", epoch, maxEpochs, currentLoss, net->totalNodes);
        }
        
        if (currentLoss < 0.0001) {
            printf("\n[Early Stopping]: Reached target threshold at epoch %d.\n", epoch);
            break;
        }
        
        for (int n = 0; n < net->totalNodes; n++) {
            mutate_node(&net->nodes[n], net->totalNodes, net->vectorSize);
        }
        
        if (((double)rand() / RAND_MAX) < 0.15) {
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
    
    printf("\n--- Safe C Graph Text Generation Output ---\n");
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

void free_network(FullyInterconnectedGraphNetwork* net) {
    for (int i = 0; i < net->nodeCapacity; i++) {
        for (int j = 0; j < net->nodeCapacity; j++) {
            free(net->nodes[i].weights[j]);
        }
        free(net->nodes[i].weights);
        free(net->nodes[i].functionLibrary);
        free(net->nodes[i].output);
    }
    free(net->nodes);
    free(net->inputNodeIds);
    free(net->outputNodeIds);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));
    FullyInterconnectedGraphNetwork net;
    
    init_network(&net, 8, 12);
    
    if (argc >= 3 && strcmp(argv[1], "--train") == 0) {
        train_from_file(&net, argv[2], 200);
    } else {
        const char* seed = "SafeRun";
        generate_text(&net, seed, 60);
    }
    
    free_network(&net);
    return 0;
}
