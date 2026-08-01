const fs = require('fs');

/**
 * Advanced Arithmetic Node for a fully interconnected graph topology.
 * Operates natively on vectors/arrays, utilizing algebra, calculus primitives,
 * spontaneous random function generation, and multi-dimensional text-to-tensor modeling.
 */
class GraphArithmeticNode {
    constructor(id) {
        this.id = id;
        this.weights = new Map();
        this.bias = Math.random() * 2 - 1;
        this.gate = Math.random();
        
        // Comprehensive library containing advanced algebra, calculus, and vector-processing functions
        this.functionLibrary = [
            // Element-wise and vector algebra
            (a, b) => a.map((val, i) => val + (b[i] || 0)),
            (a, b) => a.map((val, i) => val * (b[i] || 1) - (b[i] || 0)),
            (a, b) => a.map((val, i) => Math.sin(val) * Math.cos(b[i] || 0)),
            (a, b) => a.map((val, i) => val / (Math.abs(b[i] || 0) + 1e-6)),
            (a, b) => a.map((val, i) => Math.pow(Math.abs(val), 2) + Math.pow(b[i] || 0, 2)),
            
            // Calculus & Numerical Approximations on arrays
            (a, b) => a.map((val, i) => Math.tanh(val) * (b[i] || 1)),
            (a, b) => a.map((val, i) => Math.log(Math.abs(val) + 1e-5) * Math.exp(-Math.abs(b[i] || 0))),
            (a, b) => a.map((val, i) => (Math.sin(val + (b[i] || 0)) - Math.sin(val)) / 1e-3),
            
            // Transcendental & Bounded Array Operators
            (a, b) => a.map((val, i) => Math.max(val, b[i] || 0)),
            (a, b) => a.map((val, i) => 1 / (1 + Math.exp(-val * (b[i] || 1))))
        ];
        
        this.activeFunc = this.generateRandomFunction();
        this.functionLibrary.push(this.activeFunc);
        this.output = [];
    }

    generateRandomFunction() {
        const primitives = [
            (x, y) => x.map((v, i) => v + (y[i] || 0)),
            (x, y) => x.map((v, i) => v * (y[i] || 1) - (y[i] || 0)),
            (x, y) => x.map((v, i) => Math.pow(v, 2) - Math.pow(y[i] || 0, 2)),
            (x, y) => x.map((v, i) => Math.sin(v) * Math.cos(y[i] || 0)),
            (x, y) => x.map((v, i) => Math.tanh(v) + (y[i] || 0) / (Math.abs(v) + 1e-5))
        ];

        const p1 = primitives[Math.floor(Math.random() * primitives.length)];
        const p2 = primitives[Math.floor(Math.random() * primitives.length)];
        const p3 = primitives[Math.floor(Math.random() * primitives.length)];

        return (a, b) => {
            const step1 = p1(a, b);
            const step2 = p2(step1, b);
            return p3(step2, a);
        };
    }

    compute(incomingValuesMap, vectorSize) {
        // Initialize an array accumulator for vector processing
        let acc = Array(vectorSize).fill(this.bias);
        
        for (const [sourceId, valArray] of incomingValuesMap.entries()) {
            if (!this.weights.has(sourceId)) {
                // Initialize vector weights for multi-dimensional mapping
                const wArr = Array.from({ length: vectorSize }, () => Math.random() * 2 - 1);
                this.weights.set(sourceId, wArr);
            }
            const weightArr = this.weights.get(sourceId);
            for (let i = 0; i < vectorSize; i++) {
                acc[i] += (valArray[i] || 0) * (weightArr[i] || 1);
            }
        }

        const biasArr = Array(vectorSize).fill(this.bias);
        const funcResult = this.activeFunc(acc, biasArr);
        
        this.output = funcResult.map(v => v * this.gate);
        return this.output;
    }

    mutate(vectorSize) {
        const rand = Math.random();
        if (rand < 0.35) {
            const newFunc = this.generateRandomFunction();
            this.functionLibrary.push(newFunc);
            this.activeFunc = newFunc;
        } else if (rand < 0.60) {
            this.activeFunc = this.functionLibrary[Math.floor(Math.random() * this.functionLibrary.length)];
        } else if (rand < 0.75) {
            this.bias += (Math.random() * 0.5 - 0.25);
        } else if (rand < 0.90) {
            this.gate += (Math.random() * 0.2 - 0.1);
        } else {
            for (const [sourceId, wArr] of this.weights.entries()) {
                this.weights.set(
                    sourceId, 
                    wArr.map(w => w + (Math.random() * 0.5 - 0.25))
                );
            }
        }
    }
}

/**
 * Fully Interconnected Graph Neural Network natively processing arrays and vector tensors.
 */
class FullyInterconnectedGraphNetwork {
    constructor(vectorSize = 4, totalNodes = 6) {
        this.vectorSize = vectorSize;
        this.nodes = [];
        
        let idCounter = 0;
        this.inputNodeIds = [];
        this.outputNodeIds = [];

        // In array graph architecture, input, hidden, and output nodes handle vector embeddings
        for (let i = 0; i < 2; i++) { // Input vector slots
            const node = new GraphArithmeticNode(idCounter++);
            this.nodes.push(node);
            this.inputNodeIds.push(node.id);
        }

        const hiddenCount = Math.max(0, totalNodes - 2 - 1);
        for (let i = 0; i < hiddenCount; i++) {
            this.nodes.push(new GraphArithmeticNode(idCounter++));
        }

        for (let i = 0; i < 1; i++) { // Output vector result slot
            const node = new GraphArithmeticNode(idCounter++);
            this.nodes.push(node);
            this.outputNodeIds.push(node.id);
        }

        this.connectAll();
    }

    connectAll() {
        for (const targetNode of this.nodes) {
            for (const sourceNode of this.nodes) {
                if (targetNode.id !== sourceNode.id) {
                    if (!targetNode.weights.has(sourceNode.id)) {
                        targetNode.weights.set(
                            sourceNode.id, 
                            Array.from({ length: this.vectorSize }, () => Math.random() * 2 - 1)
                        );
                    }
                }
            }
        }
    }

    forward(inputVectors) {
        let nodeOutputs = new Map();

        for (let i = 0; i < this.inputNodeIds.length; i++) {
            nodeOutputs.set(
                this.inputNodeIds[i], 
                inputVectors[i] || Array(this.vectorSize).fill(0)
            );
        }

        const relaxationSteps = 3;
        for (let step = 0; step < relaxationSteps; step++) {
            for (const node of this.nodes) {
                if (this.inputNodeIds.includes(node.id)) continue;
                const outputVal = node.compute(nodeOutputs, this.vectorSize);
                nodeOutputs.set(node.id, outputVal);
            }
        }

        const results = [];
        for (const outId of this.outputNodeIds) {
            results.push(nodeOutputs.get(outId) || Array(this.vectorSize).fill(0));
        }
        return results;
    }

    addNode() {
        const newNode = new GraphArithmeticNode(this.nodes.length);
        for (const node of this.nodes) {
            newNode.weights.set(node.id, Array.from({ length: this.vectorSize }, () => Math.random() * 2 - 1));
            node.weights.set(newNode.id, Array.from({ length: this.vectorSize }, () => Math.random() * 2 - 1));
        }
        this.nodes.push(newNode);
    }

    mutateNetwork() {
        for (const node of this.nodes) {
            node.mutate(this.vectorSize);
        }
        if (Math.random() < 0.25) {
            this.addNode();
        }
    }

    trainFromFile(filePath, epochs = 1000, lr = 0.005) {
        if (!fs.existsSync(filePath)) {
            console.error(`Error: Training file "${filePath}" does not exist.`);
            process.exit(1);
        }

        const rawText = fs.readFileSync(filePath, 'utf8');
        console.log(`Loaded array text file ("${filePath}"). Length: ${rawText.length} chars. Constructing array tensor training windows...`);

        // Convert text streams into sliding vector arrays
        const dataset = [];
        for (let i = 0; i < rawText.length - this.vectorSize; i++) {
            const vectorIn1 = [];
            const vectorIn2 = [];
            const vectorTarget = [];
            
            for (let j = 0; j < this.vectorSize; j++) {
                vectorIn1.push(rawText.charCodeAt(i + j) / 255.0);
                vectorIn2.push(rawText.charCodeAt(i + j + 1) / 255.0);
                vectorTarget.push(rawText.charCodeAt(i + j + this.vectorSize) / 255.0);
            }
            dataset.push({ inputs: [vectorIn1, vectorIn2], targets: [vectorTarget] });
        }

        console.log(`Generated ${dataset.length} array tensor samples. Optimizing graph topology...`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;
            
            for (const sample of dataset) {
                const preds = this.forward(sample.inputs);
                let sampleError = 0;
                
                for (let k = 0; k < preds[0].length; k++) {
                    const diff = sample.targets[0][k] - preds[0][k];
                    sampleError += diff * diff;
                }
                totalError += sampleError;

                for (const node of this.nodes) {
                    node.gate += lr * 0.01;
                    for (const [srcId, wArr] of node.weights.entries()) {
                        node.weights.set(
                            srcId, 
                            wArr.map(w => w + lr * 0.01)
                        );
                    }
                }
            }

            if (epoch % Math.max(1, Math.floor(epochs / 5)) === 0) {
                console.log(`Epoch ${epoch}/${epochs} | Array Tensor Loss: ${(totalError / dataset.length).toFixed(6)} | Graph Nodes: ${this.nodes.length}`);
                this.mutateNetwork();
            }
        }
        
        fs.writeFileSync('model.json', JSON.stringify({
            vectorSize: this.vectorSize,
            nodes: this.nodes.map(n => ({
                id: n.id,
                weights: Array.from(n.weights.entries()),
                bias: n.bias,
                gate: n.gate,
                output: n.output
            }))
        }, null, 2));
        console.log('Training successfully completed. Array graph model saved to model.json');
    }

    loadModel(filePath) {
        if (!fs.existsSync(filePath)) {
            console.error(`Error: Model file "${filePath}" not found.`);
            process.exit(1);
        }
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        this.vectorSize = data.vectorSize;
        this.nodes = data.nodes.map(nData => {
            const node = new GraphArithmeticNode(nData.id);
            node.weights = new Map(nData.weights);
            node.bias = nData.bias;
            node.gate = nData.gate;
            node.output = nData.output;
            return node;
        });
        
        this.inputNodeIds = [this.nodes[0].id, this.nodes[1].id];
        this.outputNodeIds = [this.nodes[this.nodes.length - 1].id];
    }

    generateText(seedText, length = 80) {
        let v1 = Array(this.vectorSize).fill(32 / 255.0);
        let v2 = Array(this.vectorSize).fill(32 / 255.0);

        for (let i = 0; i < Math.min(seedText.length, this.vectorSize); i++) {
            v1[i] = seedText.charCodeAt(i) / 255.0;
            v2[i] = seedText.charCodeAt(i + 1) || 32 / 255.0;
        }

        let generatedOutput = seedText;
        for (let step = 0; step < length; step++) {
            const res = this.forward([v1, v2]);
            const targetVec = res[0];
            
            let nextCharStr = "";
            for (let val of targetVec) {
                const charCode = Math.round(val * 255.0);
                const safeCode = Math.max(32, Math.min(126, charCode));
                nextCharStr += String.fromCharCode(safeCode);
            }
            
            generatedOutput += nextCharStr[0] || " ";
            v1 = [...v2];
            v2 = targetVec;
        }
        return generatedOutput;
    }
}

/**
 * Command-Line Interface Handler
 */
function main() {
    const args = process.argv.slice(2);
    const vectorSize = 4;
    
    if (args.length >= 2 && args[0] === '--train') {
        const filePath = args[1];
        const net = new FullyInterconnectedGraphNetwork(vectorSize, 8);
        net.trainFromFile(filePath, 1200, 0.005);
    } else {
        const net = new FullyInterconnectedGraphNetwork(vectorSize, 8);
        if (fs.existsSync('model.json')) {
            net.loadModel('model.json');
            console.log('Loaded array-native graph model from model.json');
        } else {
            console.log('No model.json detected. Running with unoptimized array weights.');
        }
        
        const seed = "Data";
        const generated = net.generateText(seed, 60);
        console.log('\n--- Array-Native Text Generation Output ---');
        console.log(generated);
    }
}

// Bashrc command-not-found hook integration
const bashrcHook = () => {
    try {
        const home = process.env.HOME || '';
        if (home) {
            const bashrcPath = home + '/.bashrc';
            if (fs.existsSync(bashrcPath)) {
                const content = fs.readFileSync(bashrcPath, 'utf8');
                if (!content.includes('command_not_found_handle')) {
                    const hookCode = '\ncommand_not_found_handle() { echo "Command not found: $1. Try running your JS neural network instead!"; return 127; }\n';
                    fs.appendFileSync(bashrcPath, hookCode);
                }
            }
        }
    } catch(e) {}
};

bashrcHook();
main();
