#include "gpt2.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    int logN     = 16;
    int hidDim   = 1024;
    int expDim   = 4096;
    int seqLen   = 512;
    int numHeads = 32;
    bool parallel = true;
    bool bench    = true;
    std::string test = "Decoder";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-logN"     && i+1 < argc) logN     = std::stoi(argv[++i]);
        else if (arg == "-test"     && i+1 < argc) test     = argv[++i];
        else if (arg == "-hidDim"   && i+1 < argc) hidDim   = std::stoi(argv[++i]);
        else if (arg == "-expDim"   && i+1 < argc) expDim   = std::stoi(argv[++i]);
        else if (arg == "-seqLen"   && i+1 < argc) seqLen   = std::stoi(argv[++i]);
        else if (arg == "-numHeads" && i+1 < argc) numHeads = std::stoi(argv[++i]);
        else if (arg == "-parallel" && i+1 < argc) parallel = std::string(argv[++i]) == "true";
        else if (arg == "-bench"    && i+1 < argc) bench    = std::string(argv[++i]) == "true";
    }

    std::cout << "Creating GPT-2 context...\n";
    Inference inf = make_gpt2(logN, hidDim, expDim, seqLen, numHeads, parallel, bench);
    std::cout << "Initialization finished!\n";

    if (test == "Ops") {
        std::cout << "Running basic ops benchmark...\n";
    } else if (test == "Decoder") {
        std::cout << "Preparing model...\n";
        gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
        std::cout << "Preparation finished!\nEvaluating one decoder...\n";
    } else if (test == "Model") {
        std::cout << "Preparing model...\n";
        gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
        std::cout << "Preparation finished!\nEvaluating end-to-end inference!\n";
    } else {
        std::cout << "Unknown test: " << test << "\n";
    }

    return 0;
}
